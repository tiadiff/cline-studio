import { Logger } from "@/shared/services/Logger"

export interface LocalAiRecoveryOptions {
	freeVramUrl?: string
	maxAttempts?: number
	maxLoopRecoveryAttempts?: number
	windowMs?: number
	compactTask: () => Promise<void>
	sendAskResponse: (prompt: string) => Promise<void>
	isLocalProvider?: () => boolean
	isSessionRunning?: () => boolean
	waitForSessionIdle?: () => Promise<void>
}

/**
 * Automatically recovers from LLM / provider generation errors:
 * 1. Frees VRAM / unloads active model via LocalAI Gateway (http://127.0.0.1:8080/_localai/free-vram)
 * 2. Condenses / compacts context window (compactTask)
 * 3. Sends prompt "compact context and resume task" to seamlessly resume the job
 *
 * Enforces rate limiting: max 5 recovery attempts in a sliding 3-minute window
 * (and max 1 recovery attempt for repeated tool-call loop aborts so loops do not cascade).
 */
export class LocalAiRecoveryCoordinator {
	private readonly freeVramUrl: string
	private readonly maxAttempts: number
	private readonly maxLoopRecoveryAttempts: number
	private readonly windowMs: number
	private attemptTimestamps: number[] = []
	private loopAttemptTimestamps: number[] = []
	private isRecovering = false

	constructor(private readonly options: LocalAiRecoveryOptions) {
		this.freeVramUrl = options.freeVramUrl ?? "http://127.0.0.1:8080/_localai/free-vram"
		this.maxAttempts = options.maxAttempts ?? 5
		this.maxLoopRecoveryAttempts = options.maxLoopRecoveryAttempts ?? 1
		this.windowMs = options.windowMs ?? 3 * 60 * 1000 // 3 minutes
	}

	async handleTurnError(customPrompt?: string): Promise<boolean> {
		if (this.isRecovering) {
			Logger.log("[LocalAiRecovery] Recovery already in flight; ignoring duplicate error trigger")
			return false
		}

		if (this.options.isLocalProvider && !this.options.isLocalProvider()) {
			return false
		}

		const now = Date.now()
		// Clean up attempts outside sliding window (3 minutes)
		this.attemptTimestamps = this.attemptTimestamps.filter((ts) => now - ts < this.windowMs)
		this.loopAttemptTimestamps = this.loopAttemptTimestamps.filter((ts) => now - ts < this.windowMs)

		const isLoopAbort = Boolean(
			customPrompt &&
				(customPrompt.includes("consecutive identical calls") ||
					customPrompt.includes("stopping to avoid a loop")),
		)

		if (isLoopAbort && this.loopAttemptTimestamps.length >= this.maxLoopRecoveryAttempts) {
			Logger.warn(
				`[LocalAiRecovery] Loop recovery already attempted (${this.loopAttemptTimestamps.length}/${this.maxLoopRecoveryAttempts}) within ${this.windowMs / 1000}s. Halting auto-resume to prevent a runaway tool loop.`,
			)
			return false
		}

		if (this.attemptTimestamps.length >= this.maxAttempts) {
			Logger.warn(
				`[LocalAiRecovery] Max recovery attempts (${this.maxAttempts}) reached in ${this.windowMs / 1000}s. Halting automated recovery.`,
			)
			return false
		}

		this.attemptTimestamps.push(now)
		if (isLoopAbort) {
			this.loopAttemptTimestamps.push(now)
		}
		const attemptNumber = this.attemptTimestamps.length
		this.isRecovering = true

		try {
			Logger.log(`[LocalAiRecovery] Starting auto-recovery attempt ${attemptNumber}/${this.maxAttempts}...`)

			// Step 1: Wait until the active session turn has completely stopped and settled into idle/resumable state.
			// Must happen BEFORE compactTask() and sendAskResponse() because compactTask() refuses to run while
			// activeSession.isRunning is true, and send() while running queues into a dying error turn.
			if (this.options.waitForSessionIdle || this.options.isSessionRunning) {
				await new Promise((resolve) => setTimeout(resolve, 150))
				try {
					await this.options.waitForSessionIdle?.()
				} catch (waitErr) {
					Logger.warn("[LocalAiRecovery] waitForSessionIdle warning:", waitErr)
				}
				let waitCycles = 0
				while (this.options.isSessionRunning?.() && waitCycles < 40) {
					await new Promise((resolve) => setTimeout(resolve, 250))
					waitCycles++
				}
			}

			// Step 2: Free VRAM / unload model from LocalAI Gateway
			try {
				const controller = new AbortController()
				const timeout = setTimeout(() => controller.abort(), 4000)
				const res = await fetch(this.freeVramUrl, {
					method: "POST",
					signal: controller.signal,
				})
				clearTimeout(timeout)
				if (res.ok) {
					Logger.log("[LocalAiRecovery] Successfully flushed VRAM via LocalAI Gateway")
				}
			} catch (fetchErr) {
				Logger.warn("[LocalAiRecovery] Failed to notify LocalAI Gateway to free VRAM (gateway might be idle/down):", fetchErr)
			}

			// Step 3: Compact context window (non-fatal: always continue to Step 4 even if compaction is skipped or fails)
			try {
				Logger.log("[LocalAiRecovery] Compacting context window...")
				await this.options.compactTask()
			} catch (compactErr) {
				Logger.warn("[LocalAiRecovery] Compacting context window skipped or failed:", compactErr)
			}

			// Ensure session is still idle after compaction and settle state
			let postCompactCycles = 0
			while (this.options.isSessionRunning?.() && postCompactCycles < 20) {
				await new Promise((resolve) => setTimeout(resolve, 250))
				postCompactCycles++
			}
			await new Promise((resolve) => setTimeout(resolve, 500))

			// Step 4: Automatically resume task
			const resumePrompt = customPrompt ?? "compact context and resume task"
			Logger.log(`[LocalAiRecovery] Sending "${resumePrompt}"...`)
			await this.options.sendAskResponse(resumePrompt)

			Logger.log(`[LocalAiRecovery] Recovery attempt ${attemptNumber} successfully triggered.`)
			return true
		} catch (error) {
			Logger.error("[LocalAiRecovery] Recovery failed with error:", error)
			return false
		} finally {
			this.isRecovering = false
		}
	}

	reset(): void {
		this.attemptTimestamps = []
		this.loopAttemptTimestamps = []
		this.isRecovering = false
	}
}
