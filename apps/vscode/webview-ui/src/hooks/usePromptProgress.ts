import { useEffect, useState } from "react"
import { PLATFORM_CONFIG } from "@/config/platform.config"

/**
 * Polls the local AI prompt processing progress while streaming is active
 * and reasoning / prompt evaluation has not finished.
 */
export function usePromptProgress(isStreaming: boolean): number | null {
	const [progress, setProgress] = useState<number | null>(null)

	useEffect(() => {
		if (!isStreaming) {
			setProgress(null)
			return
		}

		const handleMessage = (event: MessageEvent) => {
			const msg = event.data
			if (msg?.type === "studio_prompt_progress") {
				if (typeof msg.promptProgress === "number") {
					setProgress(msg.promptProgress)
				} else {
					setProgress(null)
				}
			}
		}

		window.addEventListener("message", handleMessage)

		// Poll immediately then on interval while streaming
		PLATFORM_CONFIG.postMessage({ type: "studio_get_prompt_progress" })
		const timer = setInterval(() => {
			PLATFORM_CONFIG.postMessage({ type: "studio_get_prompt_progress" })
		}, 400)

		return () => {
			clearInterval(timer)
			window.removeEventListener("message", handleMessage)
		}
	}, [isStreaming])

	return progress
}

/**
 * Tracks live elapsed seconds while an LLM turn or reasoning step is streaming.
 */
export function useStreamingElapsedSeconds(active: boolean, startTs?: number): number {
	const [elapsedSec, setElapsedSec] = useState<number>(0)

	useEffect(() => {
		if (!active) {
			return
		}
		const baseTs = startTs && startTs > 1_000_000_000_000 ? startTs : Date.now()
		const update = () => {
			setElapsedSec(Math.max(0, Math.round((Date.now() - baseTs) / 100) / 10))
		}
		update()
		const timer = setInterval(update, 300)
		return () => clearInterval(timer)
	}, [active, startTs])

	return elapsedSec
}
