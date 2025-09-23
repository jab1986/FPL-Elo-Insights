import type {
  Player,
  Team,
  Match,
  PlayerMatchStats,
  GameweekSummary,
  UserTeamResponse,
} from '../types/fpl';

const API_BASE_URL = 'http://localhost:8001/api'; // Will be configurable later

class ApiService {
  private async fetchData<T>(endpoint: string): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error(`API Error fetching ${endpoint}:`, error);
      throw error;
    }
  }

  // Players

  async getPlayers(filters?: Record<string, string>): Promise<Player[]> {
    const queryParams = filters ? new URLSearchParams(filters).toString() : '';
    const endpoint = `/players${queryParams ? `?${queryParams}` : ''}`;
    return this.fetchData<Player[]>(endpoint);
  }

  async getPlayerById(id: number): Promise<Player> {
    return this.fetchData<Player>(`/players/${id}`);
  }

  async getTopPlayers(limit: number = 10): Promise<Player[]> {
    return this.fetchData<Player[]>(`/players/top?limit=${limit}`);
  }

  // Teams
  async getTeams(): Promise<Team[]> {
    return this.fetchData<Team[]>('/teams');
  }

  async getTeamById(id: number): Promise<Team> {
    return this.fetchData<Team>(`/teams/${id}`);
  }

  // Matches
  async getMatches(gameweek?: number): Promise<Match[]> {
    const endpoint = gameweek ? `/matches?gameweek=${gameweek}` : '/matches';
    return this.fetchData<Match[]>(endpoint);
  }

  async getMatchById(id: number): Promise<Match> {
    return this.fetchData<Match>(`/matches/${id}`);
  }

  // Player Match Stats
  async getPlayerMatchStats(matchId?: number, playerId?: number): Promise<PlayerMatchStats[]> {
    let endpoint = '/player-match-stats';
    const params = new URLSearchParams();

    if (matchId) params.append('match_id', matchId.toString());
    if (playerId) params.append('player_id', playerId.toString());

    if (params.toString()) {
      endpoint += `?${params.toString()}`;
    }

    return this.fetchData<PlayerMatchStats[]>(endpoint);
  }

  // Gameweek Summaries
  async getGameweekSummaries(): Promise<GameweekSummary[]> {
    return this.fetchData<GameweekSummary[]>('/gameweek-summaries');
  }

  async getCurrentGameweek(): Promise<GameweekSummary> {
    return this.fetchData<GameweekSummary>('/gameweek-summaries/current');
  }

  // Dashboard Stats
  async getDashboardStats(): Promise<{
    totalPlayers: number;
    totalTeams: number;
    currentGameweek: number;
    averagePoints: number;
    topScorer: Player | null;
    mostValuable: Player | null;
  }> {
    return this.fetchData('/dashboard/stats');
  }

  // Chart Data
  async getPlayerPerformanceOverTime(playerId: number): Promise<Array<{
    gameweek: number;
    points: number;
    elo: number;
    date: string;
  }>> {
    return this.fetchData(`/players/${playerId}/performance`);
  }

  async getTeamPerformanceOverTime(teamId: number): Promise<Array<{
    gameweek: number;
    points: number;
    wins: number;
    draws: number;
    losses: number;
  }>> {
    return this.fetchData(`/teams/${teamId}/performance`);
  }

  // Search
  async searchPlayers(query: string): Promise<Player[]> {
    return this.fetchData<Player[]>(`/players/search?q=${encodeURIComponent(query)}`);
  }

  async searchTeams(query: string): Promise<Team[]> {
    return this.fetchData<Team[]>(`/teams/search?q=${encodeURIComponent(query)}`);
  }

  async getUserTeam(teamId: number, event?: number): Promise<UserTeamResponse> {
    const params = event ? `?event=${event}` : '';
    return this.fetchData<UserTeamResponse>(`/user-teams/${teamId}${params}`);
  }
}

export const apiService = new ApiService();
