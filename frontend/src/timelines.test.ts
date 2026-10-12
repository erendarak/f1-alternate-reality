import { describe, expect, it } from "vitest";

import type { TimelineInfo } from "./api/types";
import { timelineTree } from "./components/Timelines";

const tl = (id: string, parent: string | null): TimelineInfo => ({
  timeline_id: id, world_id: "w", parent_timeline_id: parent, label: id, fork_snapshot_id: null,
  created_utc: "", snapshot_count: 0, fork_event_seq: null, event_count: 0, command_count: 0,
  snapshots: [],
});

describe("timelineTree", () => {
  it("puts each branch right under its parent, indented", () => {
    const tree = timelineTree([tl("main", null), tl("b2", "main"), tl("b1a", "b1"), tl("b1", "main")]);
    expect(tree.map(({ t, depth }) => `${t.timeline_id}:${depth}`)).toEqual([
      "main:0", "b2:1", "b1:1", "b1a:2",
    ]);
  });
  it("shows a branch whose parent is missing as a root instead of dropping it", () => {
    expect(timelineTree([tl("orphan", "gone")]).map(({ t }) => t.timeline_id)).toEqual(["orphan"]);
  });
});
