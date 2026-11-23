import { Logger } from "@/shared/services/Logger"

export interface LocalAiRecoveryOptions {
	freeVramUrl?: string
	maxAttempts?: number
	windowMs?: number
	compactTask: () => Promise<void>
	sendAskResponse: (prompt: string) => Promise<void>
	isLocalProvider?: () => boolean
}

/**
 * Automatically recovers from LLM / provider generation errors:
 * 1. Frees VRAM / unloads active model via LocalAI Gateway (http://127.0.0.1:8080/_localai/free-vram)
 * 2. Condenses / compacts context window (compactTask)
 * 3. Sends prompt "compact context and resume task" to seamlessly resume the job
 *
 * Enforces rate limiting: max 5 recovery attempts in a sliding 3-minute window.
 */
export class LocalAiRecoveryCoordinator {
	private readonly freeVramUrl: string
	private readonly maxAttempts: number
	private readonly windowMs: number
	private attemptTimestamps: number[] = []
	private isRecovering = false

	constructor(private readonly options: LocalAiRecoveryOptions) {
		this.freeVramUrl = options.freeVramUrl ?? "http://127.0.0.1:8080/_localai/free-vram"
		this.maxAttempts = options.maxAttempts ?? 5
		this.windowMs = options.windowMs ?? 3 * 60 * 1000 // 3 minutes
	}

	async handleTurnError(): Promise<boolean> {
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

		if (this.attemptTimestamps.length >= this.maxAttempts) {
			Logger.warn(
				`[LocalAiRecovery] Max recovery attempts (${this.maxAttempts}) reached in ${this.windowMs / 1000}s. Halting automated recovery.`,
			)
			return false
		}

		this.attemptTimestamps.push(now)
		const attemptNumber = this.attemptTimestamps.length
		this.isRecovering = true

		try {
			Logger.log(`[LocalAiRecovery] Starting auto-recovery attempt ${attemptNumber}/${this.maxAttempts}...`)

			// Step 1: Free VRAM / unload model from LocalAI Gateway
			try {
				const controller = new AbortController()
				const timeout = setTimeout(() => controller.abort(), 3000)
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

			// Step 2: Compact context window
			Logger.log("[LocalAiRecovery] Compacting context window...")
			await this.options.compactTask()

			// Short pause to ensure clean model state
			await new Promise((resolve) => setTimeout(resolve, 500))

			// Step 3: Automatically resume task
			Logger.log('[LocalAiRecovery] Sending "compact context and resume task"...')
			await this.options.sendAskResponse("compact context and resume task")

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
		this.isRecovering = false
	}
}
