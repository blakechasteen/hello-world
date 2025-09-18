/** [9 Category] Shared type universe for loom entities. */
export type Task = {
  id: string;
  name: string;
  status: string;
  tags: string[];
  totalMinutes: number;
};

export type TimeEntry = {
  id: string;
  taskId: string;
  minutes: number;
  startedAt: number;
};

export type Junction = {
  id: string;
  knotId: string;
  threadId: string;
};
