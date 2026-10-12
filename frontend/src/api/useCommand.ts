// Run a backend request with a pending state and a notice. Success is only reported from an
// accepted ack (or a successful response), never assumed (plan §13.4).

import { useCallback, useState } from "react";

import { ApiError } from "./client";
import type { CommandAck } from "./types";

export type Notice = { kind: "ok" | "error" | "pending"; text: string } | null;

export function errorText(e: unknown): string {
  return e instanceof ApiError ? `${e.code}: ${e.message}` : `backend unreachable (${String(e)})`;
}

export function useCommand() {
  const [pending, setPending] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice>(null);

  /** A command: `ok` turns an accepted ack into a message (none = clear the notice). */
  const run = useCallback(
    async (
      name: string,
      send: () => Promise<CommandAck>,
      ok?: (ack: CommandAck) => string,
      pendingText?: string,
    ): Promise<CommandAck | null> => {
      setPending(name);
      if (pendingText) setNotice({ kind: "pending", text: pendingText });
      try {
        const ack = await send();
        if (!ack.accepted) {
          setNotice({ kind: "error", text: `${ack.type} refused: ${ack.reason ?? "?"}` });
        } else {
          setNotice(ok ? { kind: "ok", text: ok(ack) } : null);
        }
        return ack;
      } catch (e) {
        setNotice({ kind: "error", text: errorText(e) });
        return null;
      } finally {
        setPending(null);
      }
    },
    [],
  );

  /** Any other request (load, fork, ...). */
  const request = useCallback(
    async <T,>(name: string, send: () => Promise<T>, ok: (r: T) => string, pendingText?: string) => {
      setPending(name);
      if (pendingText) setNotice({ kind: "pending", text: pendingText });
      try {
        const r = await send();
        setNotice({ kind: "ok", text: ok(r) });
        return r;
      } catch (e) {
        setNotice({ kind: "error", text: errorText(e) });
        return null;
      } finally {
        setPending(null);
      }
    },
    [],
  );

  return { pending, notice, setNotice, run, request };
}
