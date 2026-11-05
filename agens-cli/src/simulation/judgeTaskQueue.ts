type PendingTask<T = unknown> = {
  label: string;
  run: () => Promise<T>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
};

export interface JudgeTaskFailure {
  label: string;
  error: Error;
}

/** A per-simulation, single-consumer FIFO for turn-level judge calls. */
export class JudgeTaskQueue {
  private readonly pending: PendingTask[] = [];
  private readonly failures: JudgeTaskFailure[] = [];
  private readonly idleWaiters: Array<() => void> = [];
  private running = false;

  enqueue<T>(label: string, run: () => Promise<T>): Promise<T> {
    this.assertHealthy();
    let resolveTask!: (value: T) => void;
    let rejectTask!: (error: Error) => void;
    const result = new Promise<T>((resolve, reject) => {
      resolveTask = resolve;
      rejectTask = reject;
    });
    // Queue tasks may complete before their owner awaits them. Attach a handler
    // now so strict failures are reported through drain/assertHealthy instead
    // of becoming process-level unhandled rejections.
    void result.catch(() => undefined);
    this.pending.push({
      label,
      run,
      resolve: resolveTask as (value: unknown) => void,
      reject: rejectTask,
    });
    this.start();
    return result;
  }

  assertHealthy(): void {
    if (this.failures.length) throw this.buildFailure();
  }

  async drain(): Promise<void> {
    if (this.running || this.pending.length) {
      await new Promise<void>((resolve) => this.idleWaiters.push(resolve));
    }
    this.assertHealthy();
  }

  private start(): void {
    if (this.running) return;
    this.running = true;
    void this.pump();
  }

  private async pump(): Promise<void> {
    try {
      while (this.pending.length) {
        const task = this.pending.shift()!;
        try {
          const value = await task.run();
          task.resolve(value);
        } catch (error) {
          const normalized = error instanceof Error ? error : new Error(String(error));
          this.failures.push({ label: task.label, error: normalized });
          task.reject(new Error(`${task.label}: ${normalized.message}`, { cause: normalized }));
        }
      }
    } finally {
      this.running = false;
      for (const resolve of this.idleWaiters.splice(0)) resolve();
      // A task can be enqueued by an idle waiter before it runs.
      if (this.pending.length) this.start();
    }
  }

  private buildFailure(): AggregateError {
    return new AggregateError(
      this.failures.map(({ label, error }) => new Error(`${label}: ${error.message}`, { cause: error })),
      `Turn judge queue failed: ${this.failures
        .map(({ label, error }) => `${label}: ${error.message}`)
        .join("; ")}`,
    );
  }
}
