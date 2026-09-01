export interface Job {
  job_id: string
  name: string | null
  user: string | null
  state: string
  partition: string | null
  reason: string | null
  dependency: string | null
  exit_code: string | null
  submit_ts: number | null
  start_ts: number | null
  /** SLURM's prediction of when a queued job will start. Never a real start. */
  est_start_ts: number | null
  end_ts: number | null
  elapsed_s: number | null
  time_limit_s: number | null
  remaining_s: number | null
  cpus: string | null
  nodes: string | null
  node_list: string | null
  req_tres: string | null
  alloc_tres: string | null
  gres: string | null
  mem_req: string | null
  max_rss: string | null
  work_dir: string | null
  is_agent: boolean
  is_terminal: boolean
  is_failure: boolean
  /** A finished run the user has not opened since it finished. */
  unseen: boolean
  first_seen: number
  last_seen: number
  log_bytes: number
  log_path?: string | null
}

export interface AgentStatus {
  last_heartbeat: number | null
  seconds_since_heartbeat: number | null
  job_id: string | null
  host: string | null
  user: string | null
  version: number | null
  poll_interval: number | null
  cluster_time: number | null
}

export interface Status {
  server_time: number
  agent: AgentStatus
  state_counts: Record<string, number>
  sres: { body: string; updated_at: number } | null
  retention_days: number
}

export interface JobScript {
  job_id: string
  batch_script: string | null
  job_env: string | null
  /** Which SLURM source answered: scontrol, sacct, disk, or unavailable. */
  script_source?: string | null
  env_source?: string | null
  note?: string | null
  updated_at?: number
}

export interface LogSlice {
  job_id: string
  offset: number
  next_offset: number
  total_size: number
  text: string
}
