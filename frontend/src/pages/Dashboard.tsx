import { Link } from 'react-router-dom';
import {
  useDashboardStats,
  useCurrentGameweek,
  useTopPlayers,
  useGameweekSummaries,
} from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { TrendingUp, Users, Trophy, Target, CalendarDays, Clock } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';

const Dashboard = () => {
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError,
  } = useDashboardStats();
  const {
    data: currentGameweek,
    isLoading: gwLoading,
    error: gwError,
  } = useCurrentGameweek();
  const {
    data: topPlayers,
    isLoading: playersLoading,
    error: playersError,
  } = useTopPlayers(5);
  const {
    data: gameweekSummaries,
    isLoading: summariesLoading,
    error: summariesError,
  } = useGameweekSummaries();

  if (statsLoading || gwLoading || playersLoading || summariesLoading) {
    return <LoadingSpinner message="Loading dashboard insights..." fullHeight />;
  }

  if (statsError || gwError || playersError || summariesError) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 font-medium">Unable to load dashboard data. Please try again later.</p>
      </div>
    );
  }

  const recentGameweeks = gameweekSummaries?.slice(0, 5) ?? [];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">FPL Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Fantasy Premier League insights powered by Elo ratings and rich match data.
          </p>
        </div>
        {currentGameweek && (
          <div className="flex flex-col items-start gap-2 sm:flex-row sm:items-center sm:gap-4">
            <Badge variant="secondary" className="text-base px-4 py-2">
              Gameweek {currentGameweek.gameweek}
              {currentGameweek.finished ? ' • Finished' : ' • Active'}
            </Badge>
            <div className="flex items-center text-sm text-gray-500">
              <Clock className="h-4 w-4 mr-2" />
              Deadline: {new Date(currentGameweek.deadline_time).toLocaleString()}
            </div>
          </div>
        )}
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Players</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.totalPlayers ?? 0}</div>
            <p className="text-xs text-muted-foreground">Active FPL managers</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Teams</CardTitle>
            <Trophy className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.totalTeams ?? 0}</div>
            <p className="text-xs text-muted-foreground">Premier League clubs</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Points</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {stats?.averagePoints ? stats.averagePoints.toFixed(1) : '0.0'}
            </div>
            <p className="text-xs text-muted-foreground">Per manager this season</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Gameweek</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats?.currentGameweek ?? '--'}</div>
            <p className="text-xs text-muted-foreground">Season progress</p>
          </CardContent>
        </Card>
      </div>

      {/* Top Players */}
      <Card>
        <CardHeader>
          <CardTitle>Top Performing Players</CardTitle>
          <CardDescription>
            Highest scoring players based on total points. Click a player to view their detailed profile.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {topPlayers?.map((player, index) => (
              <div
                key={player.id}
                className="flex items-center justify-between p-4 border rounded-lg bg-white hover:border-blue-200 transition-colors"
              >
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0 w-10 h-10 bg-blue-50 rounded-full flex items-center justify-center">
                    <span className="text-sm font-semibold text-blue-600">#{index + 1}</span>
                  </div>
                  <div>
                    <Link
                      to={`/players/${player.id}`}
                      className="text-sm font-semibold text-gray-900 hover:text-blue-600"
                    >
                      {player.name}
                    </Link>
                    <p className="text-sm text-gray-500">{player.team} • {player.position}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold text-gray-900">{player.total_points}</p>
                  <p className="text-xs text-gray-500">Total points</p>
                </div>
              </div>
            ))}
            {(!topPlayers || topPlayers.length === 0) && (
              <p className="text-center text-sm text-gray-500">No player data available yet.</p>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Gameweek summaries */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Gameweek Summary</CardTitle>
          <CardDescription>Key metrics from the latest Fantasy Premier League deadlines.</CardDescription>
        </CardHeader>
        <CardContent>
          {recentGameweeks.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">
                      <div className="flex items-center">
                        <CalendarDays className="h-4 w-4 mr-2 text-gray-400" />
                        Gameweek
                      </div>
                    </th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Average Score</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Highest Score</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Deadline</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {recentGameweeks.map((gameweek) => (
                    <tr key={gameweek.gameweek} className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-medium text-gray-900">GW {gameweek.gameweek}</td>
                      <td className="px-4 py-3 text-gray-700">{gameweek.average_entry_score}</td>
                      <td className="px-4 py-3 text-gray-700">{gameweek.highest_score}</td>
                      <td className="px-4 py-3 text-gray-700">
                        {new Date(gameweek.deadline_time).toLocaleDateString()} •
                        {' '}
                        {new Date(gameweek.deadline_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td className="px-4 py-3">
                        <Badge variant={gameweek.finished ? 'default' : 'secondary'}>
                          {gameweek.finished ? 'Finished' : 'Upcoming'}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-gray-500">Gameweek summary data will appear once the season begins.</p>
          )}
        </CardContent>
      </Card>

      {/* Placeholder for upcoming enhancements */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Latest updates and insights from the FPL dataset.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <p>Recent activity and insights will be displayed here as the backend endpoints come online.</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;
