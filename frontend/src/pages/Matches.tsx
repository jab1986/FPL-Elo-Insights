import { useEffect, useMemo, useState } from 'react';
import { useMatches, useCurrentGameweek, useGameweekSummaries } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import LoadingSpinner from '../components/LoadingSpinner';

const Matches = () => {
  const { data: currentGameweek } = useCurrentGameweek();
  const { data: gameweekSummaries } = useGameweekSummaries();
  const [selectedGameweek, setSelectedGameweek] = useState<number | 'all'>('all');

  useEffect(() => {
    if (currentGameweek && selectedGameweek === 'all') {
      setSelectedGameweek(currentGameweek.gameweek);
    }
  }, [currentGameweek, selectedGameweek]);

  const gameweekOptions = useMemo(() => {
    const uniqueGameweeks = new Set<number>();
    (gameweekSummaries ?? []).forEach((summary) => uniqueGameweeks.add(summary.gameweek));
    if (currentGameweek) {
      uniqueGameweeks.add(currentGameweek.gameweek);
    }
    return Array.from(uniqueGameweeks).sort((a, b) => b - a);
  }, [gameweekSummaries, currentGameweek]);

  const {
    data: matches,
    isLoading,
    error,
  } = useMatches(selectedGameweek === 'all' ? undefined : selectedGameweek);

  if (isLoading) {
    return <LoadingSpinner message="Loading fixtures..." fullHeight />;
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 font-medium">Error loading matches data. Please try again later.</p>
      </div>
    );
  }

  const finishedMatches = matches?.filter((match) => match.finished).length ?? 0;
  const upcomingMatches = matches ? matches.length - finishedMatches : 0;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Matches</h1>
          <p className="text-gray-600 mt-1">Current and recent gameweek fixtures, with live status tracking.</p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label className="text-sm font-medium text-gray-700" htmlFor="gameweek-select">
            Gameweek
          </label>
          <select
            id="gameweek-select"
            value={selectedGameweek === 'all' ? '' : selectedGameweek}
            onChange={(event) => {
              const value = event.target.value;
              setSelectedGameweek(value === '' ? 'all' : Number.parseInt(value, 10));
            }}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="">All Gameweeks</option>
            {gameweekOptions.map((gameweek) => (
              <option key={gameweek} value={gameweek}>
                Gameweek {gameweek}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Matches</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{matches?.length ?? 0}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Finished</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{finishedMatches}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Upcoming</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{upcomingMatches}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {matches?.map((match) => (
          <Card key={match.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  {match.home_team} vs {match.away_team}
                </CardTitle>
                <Badge variant={match.finished ? 'default' : 'secondary'}>
                  {match.finished ? 'Finished' : 'Upcoming'}
                </Badge>
              </div>
              <CardDescription>
                {new Date(match.kickoff_time).toLocaleDateString()} •
                {' '}
                {new Date(match.kickoff_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {match.finished ? (
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {match.home_score} - {match.away_score}
                  </div>
                  <p className="text-sm text-gray-500 mt-2">Full-time result</p>
                </div>
              ) : (
                <div className="text-center text-gray-500">
                  <p>Match not started</p>
                  <p className="text-sm">Kickoff pending</p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {matches && matches.length === 0 && (
        <div className="text-center py-12">
          <p className="text-gray-500">No matches found for this selection.</p>
        </div>
      )}
    </div>
  );
};

export default Matches;
