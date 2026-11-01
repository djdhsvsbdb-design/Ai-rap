# RAP//AI

A standalone Russian rap lyric-writing web app using OpenRouter's free model router.

## Deploy

This project uses a Vercel serverless function at `/api/generate`.

1. Import this repository into Vercel.
2. In **Project Settings → Environment Variables**, add `OPENROUTER_API_KEY` with your OpenRouter key.
3. Redeploy the project.
4. Open the deployment URL.

Do not put the API key in `index.html`, browser JavaScript, or any committed file.

## Local development

Install the Vercel CLI, set `OPENROUTER_API_KEY` in your local environment (or an untracked `.env.local` file), then run:

```sh
npm run dev
```

The app calls `/api/generate`, which sends requests to OpenRouter's chat completions API using `openrouter/free`. Free model availability and rate limits can change.
