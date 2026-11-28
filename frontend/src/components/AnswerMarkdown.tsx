"use client";

import { useState } from "react";

interface Props {
  answer: string;
  onSelectCitation: (refId: string) => void;
}

function CodeBlock({ lang, code }: { lang: string; code: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1600);
  };

  return (
    <div className="my-3 overflow-hidden rounded-lg border border-zinc-800 bg-zinc-950">
      <div className="flex items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-3 py-1.5 text-[11px] font-mono text-zinc-400">
        <span>{lang || "snippet"}</span>
        <button
          type="button"
          onClick={handleCopy}
          className="rounded px-2 py-0.5 text-zinc-300 hover:bg-zinc-800 hover:text-white transition"
        >
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="overflow-x-auto p-3 text-xs leading-relaxed font-mono text-emerald-300">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function InlineText({ text, onSelectCitation }: { text: string; onSelectCitation: (refId: string) => void }) {
  const parts = text.split(/(\[S\d+\]|\*\*[^*]+\*\*|`[^`]+`)/g);
  return (
    <>
      {parts.map((part, idx) => {
        if (/^\[S\d+\]$/.test(part)) {
          const refId = part.slice(1, -1);
          return (
            <button
              key={idx}
              type="button"
              onClick={() => onSelectCitation(refId)}
              className="mx-0.5 inline-flex items-center rounded bg-sky-500/15 px-1.5 py-0.5 font-mono text-[11px] font-semibold text-sky-300 ring-1 ring-inset ring-sky-500/40 hover:bg-sky-500/30 transition"
            >
              {part}
            </button>
          );
        }
        if (part.startsWith("**") && part.endsWith("**")) {
          return <strong key={idx} className="font-semibold text-zinc-100">{part.slice(2, -2)}</strong>;
        }
        if (part.startsWith("`") && part.endsWith("`")) {
          return (
            <code key={idx} className="rounded bg-zinc-800/90 px-1.5 py-0.5 font-mono text-xs text-amber-200">
              {part.slice(1, -1)}
            </code>
          );
        }
        return <span key={idx}>{part}</span>;
      })}
    </>
  );
}

export function AnswerMarkdown({ answer, onSelectCitation }: Props) {
  const segments = answer.split(/(```[\s\(\S\)]*?```)/g);

  return (
    <div className="space-y-3 text-sm leading-relaxed text-zinc-200">
      {segments.map((seg, i) => {
        const trimmed = seg.trim();
        if (!trimmed) return null;

        if (trimmed.startsWith("```") && trimmed.endsWith("```")) {
          const inner = trimmed.slice(3, -3);
          const nlIdx = inner.indexOf("\n");
          const lang = nlIdx !== -1 ? inner.slice(0, nlIdx).trim() : "";
          const code = nlIdx !== -1 ? inner.slice(nlIdx + 1).trim() : inner.trim();
          return <CodeBlock key={i} lang={lang} code={code} />;
        }

        const lines = trimmed.split("\n").filter((l) => l.trim());
        const isBulletList = lines.every((l) => /^[-*]\s+/.test(l.trim()));
        if (isBulletList) {
          return (
            <ul key={i} className="list-disc space-y-1.5 pl-5 text-zinc-200">
              {lines.map((line, li) => (
                <li key={li}>
                  <InlineText text={line.replace(/^[-*]\s+/, "")} onSelectCitation={onSelectCitation} />
                </li>
              ))}
            </ul>
          );
        }

        return (
          <p key={i} className="whitespace-pre-line">
            <InlineText text={trimmed} onSelectCitation={onSelectCitation} />
          </p>
        );
      })}
    </div>
  );
}
