export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return res.status(503).json({ error: "На сервере не настроен OPENROUTER_API_KEY." });
  }

  const prompt = typeof req.body?.prompt === "string" ? req.body.prompt.trim() : "";
  if (!prompt || prompt.length > 6000) {
    return res.status(400).json({ error: "Добавь запрос длиной до 6000 символов." });
  }

  try {
    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
        "X-Title": "RAP//AI"
      },
      body: JSON.stringify({
        model: "openrouter/free",
        messages: [
          { role: "system", content: "Ты — креативный помощник по написанию оригинальных русскоязычных рэп-текстов. Пиши выразительно, соблюдай запрос пользователя и правила безопасного контента." },
          { role: "user", content: prompt }
        ],
        temperature: 0.9,
        max_tokens: 1200
      })
    });

    const data = await response.json();
    if (!response.ok) {
      return res.status(response.status).json({
        error: data?.error?.message || "OpenRouter временно не смог обработать запрос."
      });
    }

    const text = data?.choices?.[0]?.message?.content;
    if (typeof text !== "string" || !text.trim()) {
      return res.status(502).json({ error: "Модель вернула пустой ответ. Попробуй ещё раз." });
    }

    return res.status(200).json({ text: text.trim(), model: data.model || "openrouter/free" });
  } catch {
    return res.status(502).json({ error: "Не удалось связаться с OpenRouter. Попробуй позже." });
  }
}
