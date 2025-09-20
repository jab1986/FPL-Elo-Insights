import { useQuery } from '@tanstack/react-query';
import { apiService } from '../services/api';

// Players
export const usePlayers = (filters?: Record<string, string>) => {
  return useQuery({
    queryKey: ['players', filters],
    queryFn: () => apiService.getPlayers(filters),
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};

export const usePlayer = (id: number) => {
  return useQuery({
    queryKey: ['player', id],
    queryFn: () => apiService.getPlayerById(id),
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
  });
};

export const useTopPlayers = (limit: number = 10) => {
  return useQuery({
    queryKey: ['topPlayers', limit],
    queryFn: () => apiService.getTopPlayers(limit),
    staleTime: 10 * 60 * 1000, // 10 minutes
  });
};

// Teams
export const useTeams = () => {
  return useQuery({
    queryKey: ['teams'],
    queryFn: () => apiService.getTeams(),
    staleTime: 15 * 60 * 1000, // 15 minutes
  });
};

export const useTeam = (id: number) => {
  return useQuery({
    queryKey: ['team', id],
    queryFn: () => apiService.getTeamById(id),
    enabled: !!id,
    staleTime: 15 * 60 * 1000,
  });
};

// Matches
export const useMatches = (gameweek?: number) => {
  return useQuery({
    queryKey: ['matches', gameweek],
    queryFn: () => apiService.getMatches(gameweek),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
};

export const useMatch = (id: number) => {
  return useQuery({
    queryKey: ['match', id],
    queryFn: () => apiService.getMatchById(id),
    enabled: !!id,
    staleTime: 5 * 60 * 1000,
  });
};

// Player Match Stats
export const usePlayerMatchStats = (matchId?: number, playerId?: number) => {
  return useQuery({
    queryKey: ['playerMatchStats', matchId, playerId],
    queryFn: () => apiService.getPlayerMatchStats(matchId, playerId),
    enabled: !!(matchId || playerId),
    staleTime: 5 * 60 * 1000,
  });
};

// Gameweek Summaries
export const useGameweekSummaries = () => {
  return useQuery({
    queryKey: ['gameweekSummaries'],
    queryFn: () => apiService.getGameweekSummaries(),
    staleTime: 10 * 60 * 1000,
  });
};

export const useCurrentGameweek = () => {
  return useQuery({
    queryKey: ['currentGameweek'],
    queryFn: () => apiService.getCurrentGameweek(),
    staleTime: 5 * 60 * 1000,
  });
};

// Dashboard Stats
export const useDashboardStats = () => {
  return useQuery({
    queryKey: ['dashboardStats'],
    queryFn: () => apiService.getDashboardStats(),
    staleTime: 2 * 60 * 1000, // 2 minutes - more frequent updates for dashboard
  });
};

// Chart Data
export const usePlayerPerformanceOverTime = (playerId: number) => {
  return useQuery({
    queryKey: ['playerPerformance', playerId],
    queryFn: () => apiService.getPlayerPerformanceOverTime(playerId),
    enabled: !!playerId,
    staleTime: 10 * 60 * 1000,
  });
};

export const useTeamPerformanceOverTime = (teamId: number) => {
  return useQuery({
    queryKey: ['teamPerformance', teamId],
    queryFn: () => apiService.getTeamPerformanceOverTime(teamId),
    enabled: !!teamId,
    staleTime: 10 * 60 * 1000,
  });
};

// Search
export const useSearchPlayers = (query: string) => {
  return useQuery({
    queryKey: ['searchPlayers', query],
    queryFn: () => apiService.searchPlayers(query),
    enabled: query.length > 2,
    staleTime: 30 * 1000, // 30 seconds for search results
  });
};

export const useSearchTeams = (query: string) => {
  return useQuery({
    queryKey: ['searchTeams', query],
    queryFn: () => apiService.searchTeams(query),
    enabled: query.length > 2,
    staleTime: 30 * 1000,
  });
};