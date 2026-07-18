/**
 * Streamdown's math plugin only renders LaTeX wrapped in `$$` delimiters
 * (inline `$$...$$`, block `$$` on their own lines). Models frequently fall
 * back to the standard delimiters — single `$...$` for inline and
 * `\( ... \)` / `\[ ... \]` — which Streamdown then renders as plain text.
 *
 * `normalizeMathDelimiters` rewrites those variants into the `$$` form so math
 * renders regardless of which delimiters the model produced. Code spans and
 * fenced code blocks are left untouched so shell variables (`$PATH`), code
 * samples, etc. are never mangled.
 */

// Captures fenced code blocks and inline code so we can skip them. The single
// capture group means code segments land on odd indices after `split`.
const CODE_SEGMENT = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]+`)/g;

const LATEX_BLOCK = /\\\[([\s\S]+?)\\\]/g;
const LATEX_INLINE = /\\\(([\s\S]+?)\\\)/g;
// A lone `$...$` pair (not part of `$$`, not escaped) whose content starts and
// ends with a non-space character and contains no `$` or newline. Requiring
// non-space edges follows KaTeX/pandoc rules and avoids matching currency such
// as "$5 and $10" (the closing `$` there is preceded by a space).
const SINGLE_DOLLAR = /(?<![\\$])\$(?!\$)(\S(?:[^\n$]*?\S)?)\$(?!\$)/g;

function convertMath(text: string): string {
  return text
    .replace(
      LATEX_BLOCK,
      (_match, inner: string) => `\n$$\n${inner.trim()}\n$$\n`,
    )
    .replace(LATEX_INLINE, (_match, inner: string) => `$$${inner.trim()}$$`)
    .replace(SINGLE_DOLLAR, (_match, inner: string) => `$$${inner}$$`);
}

export function normalizeMathDelimiters(markdown: string): string {
  if (!markdown || (!markdown.includes("$") && !markdown.includes("\\"))) {
    return markdown;
  }
  return markdown
    .split(CODE_SEGMENT)
    .map((segment, index) => (index % 2 === 1 ? segment : convertMath(segment)))
    .join("");
}
