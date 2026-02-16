export interface MengramOptions {
  baseUrl?: string;
  timeout?: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface MemoryOptions {
  userId?: string;
  agentId?: string;
  runId?: string;
  appId?: string;
  expirationDate?: string;
}

export interface SearchOptions extends MemoryOptions {
  limit?: number;
}

export interface AddResult {
  status: string;
  message?: string;
  job_id?: string;
}

export interface SearchResult {
  entity: string;
  type: string;
  score: number;
  facts: string[];
  knowledge: any[];
  relations: any[];
}

export interface Entity {
  name: string;
  type: string;
  facts: string[];
  knowledge: any[];
  relations: any[];
  created_at: string;
  updated_at: string;
}

export interface Stats {
  entities: number;
  facts: number;
  knowledge: number;
  relations: number;
  embeddings: number;
  by_type: Record<string, number>;
}

export interface ApiKey {
  id: number;
  name: string;
  prefix: string;
  active: boolean;
  created_at: string | null;
  last_used: string | null;
}

export interface JobStatus {
  status: 'pending' | 'processing' | 'completed' | 'failed';
  result?: any;
  error?: string;
}

export interface Webhook {
  id: number;
  url: string;
  name: string;
  event_types: string[];
  active: boolean;
  trigger_count: number;
  last_triggered: string | null;
  last_error: string | null;
}

export declare class MengramError extends Error {
  statusCode: number;
  constructor(message: string, statusCode: number);
}

export declare class MengramClient {
  constructor(apiKey: string, options?: MengramOptions);

  // Memory
  add(messages: Message[], options?: MemoryOptions): Promise<AddResult>;
  addText(text: string, options?: MemoryOptions): Promise<AddResult>;
  search(query: string, options?: SearchOptions): Promise<SearchResult[]>;
  getAll(options?: MemoryOptions): Promise<Entity[]>;
  getAllFull(): Promise<Entity[]>;
  get(name: string): Promise<Entity | null>;
  delete(name: string): Promise<boolean>;
  stats(): Promise<Stats>;
  graph(): Promise<{ nodes: any[]; edges: any[] }>;
  timeline(options?: { after?: string; before?: string; limit?: number }): Promise<any[]>;

  // Agents
  runAgents(options?: { agent?: string; autoFix?: boolean }): Promise<any>;
  agentHistory(options?: { agent?: string; limit?: number }): Promise<any[]>;

  // Insights
  insights(): Promise<any>;
  reflect(): Promise<any>;

  // Webhooks
  listWebhooks(): Promise<Webhook[]>;
  createWebhook(webhook: { url: string; eventTypes: string[]; name?: string; secret?: string }): Promise<any>;
  deleteWebhook(webhookId: number): Promise<boolean>;

  // Teams
  createTeam(name: string, description?: string): Promise<any>;
  joinTeam(inviteCode: string): Promise<any>;
  listTeams(): Promise<any[]>;
  shareMemory(entityName: string, teamId: number): Promise<any>;

  // API Keys
  listKeys(): Promise<ApiKey[]>;
  createKey(name?: string): Promise<{ key: string; name: string }>;
  revokeKey(keyId: number): Promise<any>;

  // Jobs
  jobStatus(jobId: string): Promise<JobStatus>;
  waitForJob(jobId: string, options?: { pollInterval?: number; maxWait?: number }): Promise<any>;
}

export default MengramClient;
