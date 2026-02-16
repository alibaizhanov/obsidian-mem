/**
 * Mengram Cloud SDK for JavaScript / TypeScript
 *
 * Usage:
 *   const { MengramClient } = require('mengram-ai');
 *   // or: import { MengramClient } from 'mengram-ai';
 *
 *   const m = new MengramClient('mg-...');
 *
 *   await m.add([
 *     { role: 'user', content: 'I prefer dark mode and use Vim.' },
 *     { role: 'assistant', content: 'Noted!' }
 *   ], { userId: 'ali' });
 *
 *   const results = await m.search('editor preferences', { userId: 'ali' });
 */

class MengramClient {
  /**
   * @param {string} apiKey - Your Mengram API key (mg-...)
   * @param {object} [options]
   * @param {string} [options.baseUrl] - API base URL (default: https://mengram.io)
   * @param {number} [options.timeout] - Request timeout in ms (default: 30000)
   */
  constructor(apiKey, options = {}) {
    if (!apiKey) throw new Error('API key is required');
    this.apiKey = apiKey;
    this.baseUrl = (options.baseUrl || 'https://mengram.io').replace(/\/$/, '');
    this.timeout = options.timeout || 30000;
  }

  async _request(method, path, body = null, params = null) {
    let url = `${this.baseUrl}${path}`;

    if (params) {
      const qs = Object.entries(params)
        .filter(([, v]) => v !== undefined && v !== null)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
      if (qs) url += `?${qs}`;
    }

    const headers = {
      'Authorization': `Bearer ${this.apiKey}`,
      'Content-Type': 'application/json',
    };

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeout);

    try {
      const res = await fetch(url, {
        method,
        headers,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new MengramError(data.detail || `HTTP ${res.status}`, res.status);
      }
      return data;
    } catch (err) {
      if (err instanceof MengramError) throw err;
      if (err.name === 'AbortError') {
        throw new MengramError(`Request timeout after ${this.timeout}ms`, 408);
      }
      throw new MengramError(err.message, 0);
    } finally {
      clearTimeout(timer);
    }
  }

  // ---- Memory ----

  /**
   * Add memories from conversation.
   * @param {Array<{role: string, content: string}>} messages
   * @param {object} [options]
   * @param {string} [options.userId] - User ID (default: 'default')
   * @param {string} [options.agentId] - Agent ID for multi-agent systems
   * @param {string} [options.runId] - Run/session ID
   * @param {string} [options.appId] - Application ID
   * @returns {Promise<{status: string, job_id?: string}>}
   */
  async add(messages, options = {}) {
    return this._request('POST', '/v1/add', {
      messages,
      user_id: options.userId || 'default',
      agent_id: options.agentId || null,
      run_id: options.runId || null,
      app_id: options.appId || null,
    });
  }

  /**
   * Add memory from plain text.
   * @param {string} text
   * @param {object} [options]
   * @param {string} [options.userId]
   * @param {string} [options.agentId]
   * @param {string} [options.runId]
   * @param {string} [options.appId]
   * @returns {Promise<{status: string}>}
   */
  async addText(text, options = {}) {
    return this._request('POST', '/v1/add_text', {
      text,
      user_id: options.userId || 'default',
      agent_id: options.agentId || null,
      run_id: options.runId || null,
      app_id: options.appId || null,
    });
  }

  /**
   * Semantic search across memories.
   * @param {string} query
   * @param {object} [options]
   * @param {string} [options.userId]
   * @param {string} [options.agentId]
   * @param {string} [options.runId]
   * @param {string} [options.appId]
   * @param {number} [options.limit] - Max results (default: 5)
   * @returns {Promise<Array>}
   */
  async search(query, options = {}) {
    const data = await this._request('POST', '/v1/search', {
      query,
      user_id: options.userId || 'default',
      agent_id: options.agentId || null,
      run_id: options.runId || null,
      app_id: options.appId || null,
      limit: options.limit || 5,
    });
    return data.results || [];
  }

  /**
   * Get all memories.
   * @param {object} [options]
   * @param {string} [options.userId]
   * @param {string} [options.agentId]
   * @param {string} [options.appId]
   * @returns {Promise<Array>}
   */
  async getAll(options = {}) {
    const params = {};
    if (options.userId) params.user_id_param = options.userId;
    if (options.agentId) params.agent_id = options.agentId;
    if (options.appId) params.app_id = options.appId;
    const data = await this._request('GET', '/v1/memories', null, params);
    return data.memories || [];
  }

  /**
   * Get all memories with full details.
   * @returns {Promise<Array>}
   */
  async getAllFull() {
    const data = await this._request('GET', '/v1/memories/full');
    return data.memories || [];
  }

  /**
   * Get specific entity.
   * @param {string} name - Entity name
   * @returns {Promise<object|null>}
   */
  async get(name) {
    try {
      return await this._request('GET', `/v1/memory/${encodeURIComponent(name)}`);
    } catch {
      return null;
    }
  }

  /**
   * Delete a memory entity.
   * @param {string} name
   * @returns {Promise<boolean>}
   */
  async delete(name) {
    try {
      await this._request('DELETE', `/v1/entity/${encodeURIComponent(name)}`);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Get usage statistics.
   * @returns {Promise<object>}
   */
  async stats() {
    return this._request('GET', '/v1/stats');
  }

  /**
   * Get knowledge graph.
   * @returns {Promise<{nodes: Array, edges: Array}>}
   */
  async graph() {
    return this._request('GET', '/v1/graph');
  }

  /**
   * Timeline search.
   * @param {object} [options]
   * @param {string} [options.after] - ISO date
   * @param {string} [options.before] - ISO date
   * @param {number} [options.limit]
   * @returns {Promise<Array>}
   */
  async timeline(options = {}) {
    const params = { limit: options.limit || 20 };
    if (options.after) params.after = options.after;
    if (options.before) params.before = options.before;
    const data = await this._request('GET', '/v1/timeline', null, params);
    return data.results || [];
  }

  // ---- Agents ----

  /**
   * Run memory agents.
   * @param {object} [options]
   * @param {string} [options.agent] - 'curator', 'connector', 'digest', or 'all'
   * @param {boolean} [options.autoFix] - Auto-archive bad facts
   * @returns {Promise<object>}
   */
  async runAgents(options = {}) {
    return this._request('POST', '/v1/agents/run', null, {
      agent: options.agent || 'all',
      auto_fix: options.autoFix ? 'true' : 'false',
    });
  }

  /**
   * Get agent run history.
   * @param {object} [options]
   * @param {string} [options.agent]
   * @param {number} [options.limit]
   * @returns {Promise<Array>}
   */
  async agentHistory(options = {}) {
    const params = { limit: options.limit || 10 };
    if (options.agent) params.agent = options.agent;
    const data = await this._request('GET', '/v1/agents/history', null, params);
    return data.runs || [];
  }

  // ---- Insights ----

  /**
   * Get AI insights and reflections.
   * @returns {Promise<object>}
   */
  async insights() {
    return this._request('GET', '/v1/insights');
  }

  /**
   * Trigger reflection generation.
   * @returns {Promise<object>}
   */
  async reflect() {
    return this._request('POST', '/v1/reflect');
  }

  // ---- Webhooks ----

  /**
   * List webhooks.
   * @returns {Promise<Array>}
   */
  async listWebhooks() {
    const data = await this._request('GET', '/v1/webhooks');
    return data.webhooks || [];
  }

  /**
   * Create webhook.
   * @param {object} webhook
   * @param {string} webhook.url
   * @param {string[]} webhook.eventTypes
   * @param {string} [webhook.name]
   * @param {string} [webhook.secret]
   * @returns {Promise<object>}
   */
  async createWebhook(webhook) {
    return this._request('POST', '/v1/webhooks', {
      url: webhook.url,
      event_types: webhook.eventTypes,
      name: webhook.name || '',
      secret: webhook.secret || '',
    });
  }

  /**
   * Delete webhook.
   * @param {number} webhookId
   * @returns {Promise<boolean>}
   */
  async deleteWebhook(webhookId) {
    try {
      await this._request('DELETE', `/v1/webhooks/${webhookId}`);
      return true;
    } catch {
      return false;
    }
  }

  // ---- Teams ----

  /**
   * Create a team.
   * @param {string} name
   * @param {string} [description]
   * @returns {Promise<object>}
   */
  async createTeam(name, description = '') {
    return this._request('POST', '/v1/teams', { name, description });
  }

  /**
   * Join a team with invite code.
   * @param {string} inviteCode
   * @returns {Promise<object>}
   */
  async joinTeam(inviteCode) {
    return this._request('POST', '/v1/teams/join', { invite_code: inviteCode });
  }

  /**
   * List your teams.
   * @returns {Promise<Array>}
   */
  async listTeams() {
    const data = await this._request('GET', '/v1/teams');
    return data.teams || [];
  }

  /**
   * Share memory with a team.
   * @param {string} entityName
   * @param {number} teamId
   * @returns {Promise<object>}
   */
  async shareMemory(entityName, teamId) {
    return this._request('POST', `/v1/teams/${teamId}/share`, { entity: entityName });
  }

  // ---- API Keys ----

  /**
   * List API keys.
   * @returns {Promise<Array>}
   */
  async listKeys() {
    const data = await this._request('GET', '/v1/keys');
    return data.keys || [];
  }

  /**
   * Create a new API key.
   * @param {string} [name] - Key name
   * @returns {Promise<{key: string, name: string}>}
   */
  async createKey(name = 'default') {
    return this._request('POST', '/v1/keys', { name });
  }

  /**
   * Revoke an API key.
   * @param {number} keyId
   * @returns {Promise<object>}
   */
  async revokeKey(keyId) {
    return this._request('DELETE', `/v1/keys/${keyId}`);
  }

  // ---- Jobs (Async) ----

  /**
   * Check status of a background job.
   * @param {string} jobId
   * @returns {Promise<{status: string, result?: object}>}
   */
  async jobStatus(jobId) {
    return this._request('GET', `/v1/jobs/${jobId}`);
  }

  /**
   * Wait for a job to complete.
   * @param {string} jobId
   * @param {object} [options]
   * @param {number} [options.pollInterval] - ms between polls (default: 1000)
   * @param {number} [options.maxWait] - max ms to wait (default: 60000)
   * @returns {Promise<object>}
   */
  async waitForJob(jobId, options = {}) {
    const interval = options.pollInterval || 1000;
    const maxWait = options.maxWait || 60000;
    const start = Date.now();

    while (Date.now() - start < maxWait) {
      const job = await this.jobStatus(jobId);
      if (job.status === 'completed' || job.status === 'failed') {
        return job;
      }
      await new Promise(r => setTimeout(r, interval));
    }
    throw new MengramError('Job timed out', 408);
  }
}

class MengramError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'MengramError';
    this.statusCode = statusCode;
  }
}

// Export for both CommonJS and ESM
module.exports = { MengramClient, MengramError };
module.exports.default = MengramClient;
