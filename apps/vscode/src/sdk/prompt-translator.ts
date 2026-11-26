import { Logger } from "@shared/services/Logger"

export interface TranslationResult {
	translated: string
	detectedLang: string
	wasTranslated: boolean
}

/**
 * Free Google Translate client using public endpoints with fallback.
 * Automatically translates non-English prompts (especially Italian) to English,
 * preserving code blocks, markdown tags, backticks, and URLs.
 */
export class PromptTranslatorService {
	private static instance: PromptTranslatorService
	private readonly requestTimeoutMs = 6000

	private constructor() {}

	public static get(): PromptTranslatorService {
		if (!PromptTranslatorService.instance) {
			PromptTranslatorService.instance = new PromptTranslatorService()
		}
		return PromptTranslatorService.instance
	}

	/**
	 * Translates a prompt to English if it is in Italian or another non-English language.
	 * Returns the translated prompt or the original prompt if already in English or upon network failure.
	 */
	public async translatePrompt(prompt: string): Promise<TranslationResult> {
		const trimmed = prompt.trim()
		if (!trimmed) {
			return { translated: prompt, detectedLang: "en", wasTranslated: false }
		}

		// Skip internal or synthetic prompts
		if (this.isInternalOrSyntheticPrompt(trimmed)) {
			return { translated: prompt, detectedLang: "en", wasTranslated: false }
		}

		try {
			const { protectedText, placeholders } = this.protectContent(prompt)

			// Fast check: if no actual translatable characters (e.g. only symbols or numbers), skip
			if (!/[a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]/.test(protectedText)) {
				return { translated: prompt, detectedLang: "en", wasTranslated: false }
			}

			const translation = await this.fetchWithFallback(protectedText)
			if (!translation) {
				return { translated: prompt, detectedLang: "unknown", wasTranslated: false }
			}

			const detectedLang = (translation.detectedLang || "").toLowerCase()
			const isEnglish = detectedLang === "en" || detectedLang === "eng"

			if (isEnglish) {
				return { translated: prompt, detectedLang: "en", wasTranslated: false }
			}

			const restoredTranslation = this.restorePlaceholders(translation.translatedText, placeholders)
			Logger.log(
				`[PromptTranslator] Successfully translated prompt from '${detectedLang}' to 'en' (${prompt.length} -> ${restoredTranslation.length} chars)`,
			)

			return {
				translated: restoredTranslation,
				detectedLang,
				wasTranslated: true,
			}
		} catch (error) {
			Logger.warn("[PromptTranslator] Failed to translate prompt, using original:", error)
			return { translated: prompt, detectedLang: "unknown", wasTranslated: false }
		}
	}

	/**
	 * Checks if the prompt is an internal orchestration or task resumption instruction.
	 */
	private isInternalOrSyntheticPrompt(prompt: string): boolean {
		return (
			prompt.startsWith("[TASK RESUMPTION]") ||
			prompt.startsWith("[ORCHESTRATOR") ||
			prompt.startsWith("<mode_notice>") ||
			prompt.startsWith("<orchestrator_evaluation>") ||
			prompt.startsWith("<hook_context") ||
			prompt.startsWith("compact context and resume task")
		)
	}

	/**
	 * Protects fenced code blocks, inline code, URLs, and XML/HTML-like tags from being modified by translation.
	 */
	private protectContent(text: string): { protectedText: string; placeholders: string[] } {
		const placeholders: string[] = []
		const push = (m: string) => {
			placeholders.push(m)
			return `__PT_PH_${placeholders.length - 1}__`
		}

		// 1. Fenced code blocks ```...```
		let s = text.replace(/```[\s\S]*?```/g, (m) => push(m))

		// 2. Inline code `...`
		s = s.replace(/`[^`\n]+`/g, (m) => push(m))

		// 3. URLs
		s = s.replace(/https?:\/\/[^\s)]+/g, (m) => push(m))

		// 4. XML / HTML tags like <tag attr="val">...</tag>
		s = s.replace(/<\/?[a-zA-Z_0-9]+(?:\s+[^>]*)?>/g, (m) => push(m))

		return { protectedText: s, placeholders }
	}

	/**
	 * Restores placeholders after translation, tolerating spacing added by translation services.
	 */
	private restorePlaceholders(translatedText: string, placeholders: string[]): string {
		let result = translatedText
		for (let i = 0; i < placeholders.length; i++) {
			const rx = new RegExp(`__PT_PH_\\s*${i}\\s*__`, "g")
			result = result.replace(rx, () => placeholders[i])
		}
		return result
	}

	/**
	 * Calls Google Translate endpoints with automatic fallback.
	 */
	private async fetchWithFallback(
		text: string,
	): Promise<{ translatedText: string; detectedLang: string } | null> {
		// Endpoint 1: Google Clients5 dict-chrome-ex
		try {
			const url =
				"https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=auto&tl=en&q=" +
				encodeURIComponent(text)
			const res = await this.fetchWithTimeout(url)
			if (res.ok) {
				const data = (await res.json()) as any
				if (Array.isArray(data) && Array.isArray(data[0])) {
					const translatedText = typeof data[0][0] === "string" ? data[0][0] : ""
					const detectedLang = typeof data[0][1] === "string" ? data[0][1] : "auto"
					if (translatedText) {
						return { translatedText, detectedLang }
					}
				}
			}
		} catch (error) {
			Logger.warn("[PromptTranslator] Primary endpoint (clients5) failed, trying fallback:", error)
		}

		// Endpoint 2: Googleapis GTX single
		try {
			const url =
				"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q=" +
				encodeURIComponent(text)
			const res = await this.fetchWithTimeout(url)
			if (res.ok) {
				const data = (await res.json()) as any
				if (Array.isArray(data) && Array.isArray(data[0])) {
					let translatedText = ""
					for (const segment of data[0]) {
						if (segment && typeof segment[0] === "string") {
							translatedText += segment[0]
						}
					}
					const detectedLang = typeof data[2] === "string" ? data[2] : "auto"
					if (translatedText) {
						return { translatedText, detectedLang }
					}
				}
			}
		} catch (error) {
			Logger.warn("[PromptTranslator] Fallback endpoint (googleapis) failed:", error)
		}

		return null
	}

	private async fetchWithTimeout(url: string): Promise<Response> {
		const controller = new AbortController()
		const timer = setTimeout(() => controller.abort(), this.requestTimeoutMs)
		try {
			return await fetch(url, {
				headers: {
					"User-Agent":
						"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
				},
				signal: controller.signal,
			})
		} finally {
			clearTimeout(timer)
		}
	}
}
