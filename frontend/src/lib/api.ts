const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Source {
  source_number: number;
  doc_title: string;
  section: string;
  text: string;
  relevance_score: number;
  doc_id: string;
  source_url: string;
}

export interface ChatRequest {
  query: string;
  conversation_id?: string;
  top_k?: number;
}

export interface ChatResponse {
  response: string;
  sources: Source[];
  conversation_id: string;
  query: string;
}

export interface SearchRequest {
  query: string;
  top_k?: number;
  filter_doc_id?: string;
  filter_section?: string;
}

export interface SearchResult {
  text: string;
  doc_title: string;
  section: string;
  doc_id: string;
  source_url: string;
  relevance_score: number;
  chunk_index?: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total_results: number;
}

export interface HealthResponse {
  status: string;
  service: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      error.detail || `HTTP error ${response.status}`
    );
  }
  return response.json();
}

export async function sendChatMessage(
  request: ChatRequest
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  return handleResponse<ChatResponse>(response);
}

export async function searchDocuments(
  request: SearchRequest
): Promise<SearchResponse> {
  const response = await fetch(`${API_BASE_URL}/api/search`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  return handleResponse<SearchResponse>(response);
}

export async function checkHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/health`);
  return handleResponse<HealthResponse>(response);
}

export async function getChatStats(): Promise<{
  status: string;
  collection: { name: string; count: number };
}> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stats`);
  return handleResponse(response);
}

// Streaming types
export interface StreamEvent {
  type: "sources" | "chunk" | "done" | "error";
  sources?: Source[];
  conversation_id?: string;
  content?: string;
  trace_id?: string;
  message?: string;
}

export interface StreamCallbacks {
  onSources: (sources: Source[], conversationId: string) => void;
  onChunk: (content: string) => void;
  onDone: (traceId: string) => void;
  onError: (error: string) => void;
}

export async function sendChatMessageStream(
  request: ChatRequest,
  callbacks: StreamCallbacks
): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new ApiError(
      response.status,
      error.detail || `HTTP error ${response.status}`
    );
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("No response body");
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE messages
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data) {
            try {
              const event: StreamEvent = JSON.parse(data);

              switch (event.type) {
                case "sources":
                  callbacks.onSources(
                    event.sources || [],
                    event.conversation_id || ""
                  );
                  break;
                case "chunk":
                  callbacks.onChunk(event.content || "");
                  break;
                case "done":
                  callbacks.onDone(event.trace_id || "");
                  break;
                case "error":
                  callbacks.onError(event.message || "Unknown error");
                  break;
              }
            } catch {
              // Ignore JSON parse errors for incomplete chunks
            }
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}
