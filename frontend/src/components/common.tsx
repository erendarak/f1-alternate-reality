// Small shared UI pieces: modal dialog, source-type tag, notice line.

import { useEffect, useRef, type ReactNode } from "react";

import type { Notice } from "../api/useCommand";
import type { SourceType } from "../api/types";

/** Native <dialog>: focus is trapped, Esc closes, the page behind is inert. */
export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    // Closing is driven by React state only. Esc fires `cancel` synchronously: stop the browser
    // from closing the element itself and unmount it instead. (Relying on the async `close`
    // event failed in the browser pane: it never fired, leaving a closed dialog mounted that
    // couldn't be reopened.)
    const handle = (e: Event) => {
      e.preventDefault();
      onCloseRef.current();
    };
    d.addEventListener("cancel", handle);
    // no close() in the cleanup: StrictMode would close the dialog right after opening it;
    // removing the element on unmount ends the modal state anyway
    if (!d.open) d.showModal();
    // showModal() focuses the first focusable element (the ✕ button); start in the form instead
    d.querySelector<HTMLElement>(".dialog-body input, .dialog-body select, .dialog-body textarea")?.focus();
    return () => d.removeEventListener("cancel", handle);
  }, []);
  return (
    <dialog ref={ref} className="modal" aria-label={title}>
      <header>
        <h2>{title}</h2>
        <button className="ghost" onClick={onClose} aria-label="Close">
          ✕
        </button>
      </header>
      {children}
    </dialog>
  );
}

const SOURCE: Record<SourceType, { label: string; title: string }> = {
  historical_verified: {
    label: "Verified history",
    title: "Checkable record from a named source (Jolpica / F1 timing via FastF1)",
  },
  estimated_from_data: {
    label: "Estimated from data",
    title: "Derived from real 2024 race data by a fit or model; an estimate, not a record",
  },
  editorial_assumption: {
    label: "Editorial assumption",
    title: "A design choice with no measurement behind it; not a claim about the real person",
  },
  simulated_dynamic: {
    label: "Simulated",
    title: "Produced by this simulation as it runs; not what happened in the real race",
  },
  learned_parameter: { label: "Learned", title: "Produced by ML training" },
};

export function SourceTag({ source, note }: { source: SourceType; note?: string }) {
  const s = SOURCE[source];
  return (
    <span className={`source source-${source}`} title={note ? `${s.title}\n${note}` : s.title}>
      {s.label}
    </span>
  );
}

export function NoticeLine({ notice, fallback }: { notice: Notice; fallback?: string }) {
  return (
    <div className={`notice ${notice?.kind ?? ""}`} role="status" aria-live="polite">
      {notice?.text ?? fallback ?? ""}
    </div>
  );
}

export function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) ||
    document.querySelector("dialog[open]") !== null
  );
}
