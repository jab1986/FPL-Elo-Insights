import { useEffect, useMemo, useState, type FormEvent } from 'react';
import { AlertCircle, RefreshCcw, Trophy, Users, Star } from 'lucide-react';
import { useUserTeam } from '../hooks/useFPLData';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import LoadingSpinner from '../components/LoadingSpinner';
import type { UserTeamPick, UserTeamHistoryEntry } from '../types/fpl';

const DEFAULT_TEAM_ID = 266343;
const STORAGE_KEY = 'fpl-insights.team-id';

interface ChipInfo {
  name?: string;
  status_for_entry?: string;
  played_by_entry?: boolean;
  event?: number | null;
}

const formatNumber = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return value.toLocaleString();
};

const formatCurrency = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return `£${value.toFixed(1)}m`;
};

const formatPercent = (value?: number | null) => {
  if (value == null || Number.isNaN(value)) {
    return '—';
  }
  return `${value.toFixed(1)}%`;
};

const formatDate = (isoDate?: string | null) => {
  if (!isoDate) {
    return '—';
  }
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) {
    return '—';
  }
  return parsed.toLocaleDateString();
};

const MyTeam = () => {
  const [teamIdInput, setTeamIdInput] = useState(() => {
    if (typeof window === 'undefined') {
      return DEFAULT_TEAM_ID.toString();
    }
    return window.localStorage.getItem(STORAGE_KEY) ?? DEFAULT_TEAM_ID.toString();
  });

  const [activeTeamId, setActiveTeamId] = useState<number | null>(() => {
    if (typeof window === 'undefined') {
      return DEFAULT_TEAM_ID;
    }
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return DEFAULT_TEAM_ID;
    }
    const parsed = Number.parseInt(stored, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TEAM_ID;
  });

  const [inputError, setInputError] = useState<string | null>(null);

  const {
    data,
    isLoading,
    isFetching,
    error,
    refetch,
  } = useUserTeam(activeTeamId ?? undefined);

  useEffect(() => {
    if (typeof window === 'undefined' || !activeTeamId) {
      return;
    }
    window.localStorage.setItem(STORAGE_KEY, activeTeamId.toString());
  }, [activeTeamId]);

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const trimmed = teamIdInput.trim();
    if (!trimmed) {
      setInputError('Please enter your FPL team ID.');
      return;
    }
    const parsed = Number.parseInt(trimmed, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      setInputError('Team IDs must be positive numbers.');
      return;
    }
    setInputError(null);
    setActiveTeamId(parsed);
  };

  const startingXI = useMemo(() => {
    if (!data?.picks) {
      return [] as UserTeamPick[];
    }
    return [...data.picks].sort(
      (a, b) => (a.position ?? 0) - (b.position ?? 0)
    );
  }, [data?.picks]);

  const bench = useMemo(() => {
    if (data?.bench && data.bench.length > 0) {
      return [...data.bench].sort(
        (a, b) => (a.position ?? 0) - (b.position ?? 0)
      );
    }
    if (!data?.picks) {
      return [] as UserTeamPick[];
    }
    return [...data.picks]
      .filter((pick) => (pick.position ?? 0) > 11)
      .sort((a, b) => (a.position ?? 0) - (b.position ?? 0));
  }, [data?.bench, data?.picks]);

  const recentHistory = useMemo(() => {
    if (!data?.history?.current) {
      return [] as UserTeamHistoryEntry[];
    }
    return [...data.history.current]
      .filter((entry) => entry.event != null)
      .sort((a, b) => (b.event ?? 0) - (a.event ?? 0))
      .slice(0, 6);
  }, [data?.history?.current]);

  const managerName = useMemo(() => {
    const first = data?.team?.player_first_name ?? '';
    const last = data?.team?.player_last_name ?? '';
    return `${first} ${last}`.trim();
  }, [data?.team?.player_first_name, data?.team?.player_last_name]);

  const currentEventSummary = data?.current_event_summary;
  const showInitialSpinner = isLoading && !data;
  const isRefreshing = isFetching && !isLoading;
  const queryError = error instanceof Error ? error.message : null;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-3xl font-bold text-gray-900">My FPL Team</h1>
        <p className="text-gray-600">
          Add your Fantasy Premier League manager ID to keep track of your squad and
          review weekly performance.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Link your team</CardTitle>
          <CardDescription>
            Your manager ID appears in the URL on fantasy.premierleague.com. Example:{' '}
            <span className="font-semibold text-gray-700">266343</span>.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form
            onSubmit={handleSubmit}
            className="flex flex-col gap-4 sm:flex-row sm:items-end"
          >
            <div className="flex-1">
              <label htmlFor="teamId" className="block text-sm font-medium text-gray-700">
                Team ID
              </label>
              <input
                id="teamId"
                name="teamId"
                type="text"
                inputMode="numeric"
                value={teamIdInput}
                onChange={(event) => setTeamIdInput(event.target.value)}
                className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 shadow-sm focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-200"
                placeholder="e.g. 266343"
              />
              {inputError ? (
                <p className="mt-2 text-sm text-red-600">{inputError}</p>
              ) : (
                <p className="mt-2 text-xs text-gray-500">
                  Saving updates your browser so the app remembers your team next time.
                </p>
              )}
            </div>
            <div className="flex items-center gap-3">
              <button
                type="submit"
                className="inline-flex items-center rounded-md bg-blue-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
              >
                <Users className="mr-2 h-4 w-4" /> Save team
              </button>
              {activeTeamId ? (
                <button
                  type="button"
                  onClick={() => refetch()}
                  className="inline-flex items-center rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 shadow-sm transition hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1"
                >
                  <RefreshCcw className="mr-2 h-4 w-4" /> Refresh
                </button>
              ) : null}
            </div>
          </form>
        </CardContent>
      </Card>

      {data?.source === 'sample' && (
        <div className="flex items-start gap-3 rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          <Star className="mt-0.5 h-5 w-5 text-amber-500" />
          <p>
            Live FPL data was unavailable, so a curated example for team{' '}
            <span className="font-semibold">266343</span> is shown. Try refreshing when you
            have an active internet connection to load your latest squad.
          </p>
        </div>
      )}

      {queryError && !data && (
        <div className="flex items-start gap-3 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          <AlertCircle className="mt-0.5 h-5 w-5" />
          <div>
            <p className="font-semibold">Unable to load team</p>
            <p>{queryError}</p>
          </div>
        </div>
      )}

      {showInitialSpinner && (
        <LoadingSpinner message="Fetching your FPL team..." fullHeight />
      )}

      {!showInitialSpinner && data && (
        <div className="space-y-6">
          <Card>
            <CardHeader className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div>
                <CardTitle>{data.team?.name ?? 'Your fantasy squad'}</CardTitle>
                <CardDescription>
                  Managed by {managerName || '—'}
                  {data.team?.player_region_name ? ` • ${data.team.player_region_name}` : ''}
                </CardDescription>
                {data.team?.joined_time && (
                  <p className="mt-1 text-xs text-gray-500">
                    Joined: {formatDate(data.team.joined_time)}
                  </p>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-2">
                {typeof data.current_event === 'number' && (
                  <Badge variant="secondary">GW {data.current_event}</Badge>
                )}
                {data.team?.favourite_team_name && (
                  <Badge variant="outline">Favourite club: {data.team.favourite_team_name}</Badge>
                )}
                {isRefreshing && <Badge variant="secondary">Refreshing…</Badge>}
                <Badge variant={data.source === 'live' ? 'default' : 'secondary'}>
                  {data.source === 'live' ? 'Live data' : 'Sample data'}
                </Badge>
              </div>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                <div className="rounded-lg border border-gray-200 p-4">
                  <div className="flex items-center justify-between text-sm text-gray-500">
                    <span>Overall points</span>
                    <Trophy className="h-4 w-4 text-yellow-500" />
                  </div>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatNumber(data.team?.summary_overall_points)}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm text-gray-500">Overall rank</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatNumber(data.team?.summary_overall_rank)}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm text-gray-500">Gameweek points</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatNumber(currentEventSummary?.points ?? data.team?.summary_event_points)}
                  </p>
                  <p className="text-xs text-gray-500">
                    Rank: {formatNumber(currentEventSummary?.rank ?? data.team?.summary_event_rank)}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm text-gray-500">Transfers (GW)</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatNumber(
                      currentEventSummary?.event_transfers ?? data.team?.summary_event_transfers
                    )}
                  </p>
                  <p className="text-xs text-gray-500">
                    Cost: {formatNumber(
                      currentEventSummary?.event_transfers_cost ??
                        data.team?.summary_event_transfers_cost
                    )}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm text-gray-500">Squad value</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatCurrency(data.team?.team_value)}
                  </p>
                </div>
                <div className="rounded-lg border border-gray-200 p-4">
                  <p className="text-sm text-gray-500">In the bank</p>
                  <p className="mt-1 text-2xl font-semibold text-gray-900">
                    {formatCurrency(currentEventSummary?.bank ?? data.team?.bank)}
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Starting XI</CardTitle>
                <CardDescription>The players locked in for this gameweek.</CardDescription>
              </CardHeader>
              <CardContent>
                {startingXI.length > 0 ? (
                  <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                    {startingXI.map((pick) => {
                      const key = `${pick.element}-${pick.position}`;
                      return (
                        <div
                          key={key}
                          className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <p className="text-base font-semibold text-gray-900">
                                {pick.player.web_name ?? 'Unknown player'}
                              </p>
                              <p className="text-sm text-gray-500">
                                {pick.player.team ?? '—'} • {pick.player.position ?? '—'}
                              </p>
                            </div>
                            <div className="flex gap-1">
                              {pick.is_captain && <Badge variant="default">C</Badge>}
                              {pick.is_vice_captain && <Badge variant="secondary">VC</Badge>}
                            </div>
                          </div>
                          <div className="mt-4 grid grid-cols-2 gap-3 text-sm text-gray-600">
                            <div>
                              <p className="font-medium text-gray-700">Total points</p>
                              <p>{formatNumber(pick.player.total_points)}</p>
                            </div>
                            <div>
                              <p className="font-medium text-gray-700">Price</p>
                              <p>{formatCurrency(pick.player.now_cost)}</p>
                            </div>
                            <div>
                              <p className="font-medium text-gray-700">Selected by</p>
                              <p>{formatPercent(pick.player.selected_by_percent)}</p>
                            </div>
                            <div>
                              <p className="font-medium text-gray-700">Multiplier</p>
                              <p>{pick.multiplier ?? 0}x</p>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No starting lineup available.</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Bench</CardTitle>
                <CardDescription>Players ready to step in if needed.</CardDescription>
              </CardHeader>
              <CardContent>
                {bench.length > 0 ? (
                  <div className="space-y-3">
                    {bench.map((pick) => {
                      const key = `${pick.element}-${pick.position}`;
                      return (
                        <div
                          key={key}
                          className="rounded-md border border-gray-200 bg-white p-3 shadow-sm"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-sm font-semibold text-gray-900">
                                {pick.player.web_name ?? 'Unknown player'}
                              </p>
                              <p className="text-xs text-gray-500">
                                {pick.player.team ?? '—'} • {pick.player.position ?? '—'}
                              </p>
                            </div>
                            <span className="text-xs font-medium text-gray-500">
                              Slot {pick.position}
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">Bench information is not available.</p>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Recent gameweeks</CardTitle>
                <CardDescription>Last six deadlines and their outcomes.</CardDescription>
              </CardHeader>
              <CardContent>
                {recentHistory.length > 0 ? (
                  <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200 text-sm">
                      <thead className="bg-gray-50">
                        <tr>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">GW</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Points</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Rank</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Overall</th>
                          <th className="px-4 py-2 text-left font-semibold text-gray-700">Bench</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-100">
                        {recentHistory.map((entry, index) => (
                          <tr key={entry.event ?? `event-${index}`} className="bg-white">
                            <td className="px-4 py-2 font-medium text-gray-900">
                              GW {formatNumber(entry.event)}
                            </td>
                            <td className="px-4 py-2 text-gray-700">
                              {formatNumber(entry.points)}
                            </td>
                            <td className="px-4 py-2 text-gray-700">
                              {formatNumber(entry.rank)}
                            </td>
                            <td className="px-4 py-2 text-gray-700">
                              {formatNumber(entry.overall_rank)}
                            </td>
                            <td className="px-4 py-2 text-gray-700">
                              {formatNumber(entry.points_on_bench)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No gameweek history available.</p>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Chip usage</CardTitle>
                <CardDescription>Track when special chips were used or remain.</CardDescription>
              </CardHeader>
              <CardContent>
                {data.chips && data.chips.length > 0 ? (
                  <div className="space-y-3">
                    {data.chips.map((chip, index) => {
                      const chipInfo = chip as ChipInfo;
                      const name = chipInfo.name?.replace(/_/g, ' ') ?? 'Unknown chip';
                      const status = chipInfo.status_for_entry ?? 'available';
                      const played = chipInfo.played_by_entry ?? false;
                      const event = chipInfo.event;

                      let helperText = '';
                      if (played && event != null) {
                        helperText = `Played in GW ${event}`;
                      } else if (played) {
                        helperText = 'Already played';
                      } else if (event != null) {
                        helperText = `Scheduled for GW ${event}`;
                      } else {
                        helperText = 'Available to use';
                      }

                      return (
                        <div
                          key={`${name}-${index}`}
                          className="rounded-md border border-gray-200 bg-white p-3 shadow-sm"
                        >
                          <p className="text-sm font-semibold capitalize text-gray-900">{name}</p>
                          <p className="text-xs uppercase text-gray-500">{status}</p>
                          <p className="mt-1 text-sm text-gray-600">{helperText}</p>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500">No chip information available.</p>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
};

export default MyTeam;

