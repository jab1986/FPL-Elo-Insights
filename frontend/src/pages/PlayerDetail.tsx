import { Link, useParams } from 'react-router-dom';
import { usePlayer, usePlayerPerformanceOverTime, usePlayerMatchStats } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ArrowLeft } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';
import PlayerPerformanceChart from '../components/charts/PlayerPerformanceChart';

const PlayerDetail = () => {
  const { id } = useParams<{ id: string }>();
  const playerId = id ? Number.parseInt(id, 10) : 0;

  const {
    data: player,
    isLoading,
    error,
  } = usePlayer(playerId);
  const {
    data: performance,
    isLoading: performanceLoading,
    error: performanceError,
  } = usePlayerPerformanceOverTime(playerId);
  const {
    data: matchStats,
    isLoading: matchStatsLoading,
    error: matchStatsError,
  } = usePlayerMatchStats(undefined, playerId);

  if (isLoading) {
    return <LoadingSpinner message="Loading player profile..." fullHeight />;
  }

  if (error || !player) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 font-medium">Player not found.</p>
        <Link to="/players" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
          ← Back to Players
        </Link>
      </div>
    );
  }

  const recentMatches = matchStats
    ?.slice()
    .sort((a, b) => b.match_id - a.match_id)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/players" className="text-blue-600 hover:text-blue-800">
            <ArrowLeft className="h-6 w-6" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{player.name}</h1>
            <p className="text-gray-600">{player.team}</p>
          </div>
        </div>
        <Badge variant="secondary" className="w-fit px-4 py-2 text-sm font-semibold">
          {player.position}
        </Badge>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Total Points</CardTitle>
            <CardDescription>Season-to-date</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{player.total_points}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Price</CardTitle>
            <CardDescription>Current market value</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">£{(player.now_cost / 10).toFixed(1)}m</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Selected By</CardTitle>
            <CardDescription>Percent of FPL managers</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{player.selected_by_percent}%</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Points per Game</CardTitle>
            <CardDescription>Average output</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{player.points_per_game}</div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Season Statistics</CardTitle>
          <CardDescription>All key contributions recorded so far.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Minutes Played</p>
              <p className="text-lg font-semibold">{player.minutes}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Goals</p>
              <p className="text-lg font-semibold">{player.goals_scored}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Assists</p>
              <p className="text-lg font-semibold">{player.assists}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Clean Sheets</p>
              <p className="text-lg font-semibold">{player.clean_sheets}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Yellow Cards</p>
              <p className="text-lg font-semibold">{player.yellow_cards}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Red Cards</p>
              <p className="text-lg font-semibold">{player.red_cards}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Saves</p>
              <p className="text-lg font-semibold">{player.saves}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Bonus Points</p>
              <p className="text-lg font-semibold">{player.bonus}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Over Time</CardTitle>
          <CardDescription>Points and Elo rating progression across gameweeks.</CardDescription>
        </CardHeader>
        <CardContent>
          {performanceLoading ? (
            <LoadingSpinner />
          ) : performanceError ? (
            <p className="text-sm text-red-600">Unable to load performance data.</p>
          ) : performance && performance.length > 0 ? (
            <PlayerPerformanceChart data={performance} />
          ) : (
            <p className="text-sm text-gray-500">Performance data will appear once matches are available.</p>
          )}
        </CardContent>
      </Card>

      {/* Recent Matches */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Matches</CardTitle>
          <CardDescription>Game-by-game Fantasy output and involvement.</CardDescription>
        </CardHeader>
        <CardContent>
          {matchStatsLoading ? (
            <LoadingSpinner />
          ) : matchStatsError ? (
            <p className="text-sm text-red-600">Unable to load recent match data.</p>
          ) : recentMatches && recentMatches.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Match</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Minutes</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Goals</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Assists</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Points</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {recentMatches.map((match) => (
                    <tr key={match.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium text-gray-900">#{match.match_id}</td>
                      <td className="px-4 py-3 text-gray-700">{match.minutes}</td>
                      <td className="px-4 py-3 text-gray-700">{match.goals_scored}</td>
                      <td className="px-4 py-3 text-gray-700">{match.assists}</td>
                      <td className="px-4 py-3 text-gray-700">{match.total_points}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-gray-500">Match-by-match data will appear once the season is underway.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default PlayerDetail;
