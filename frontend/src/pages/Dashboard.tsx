import { useDashboardStats, useCurrentGameweek, useTopPlayers } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/Card';
import { Badge } from './ui/Badge';
import { TrendingUp, Users, Trophy, Target } from 'lucide-react';

const Dashboard = () => {
  const { data: stats, isLoading: statsLoading } = useDashboardStats();
  const { data: currentGameweek, isLoading: gwLoading } = useCurrentGameweek();
  const { data: topPlayers, isLoading: playersLoading } = useTopPlayers(5);

  if (statsLoading || gwLoading || playersLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">FPL Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Fantasy Premier League insights and analytics
          </p>
        </div>
        {currentGameweek && (
          <Badge variant="secondary" className="text-lg px-4 py-2">
            Gameweek {currentGameweek.gameweek}
            {currentGameweek.finished ? ' (Finished)' : ' (Active)'}
          </Badge>
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
            <div className="text-2xl font-bold">{stats?.totalPlayers || 0}</div>
            <p className="text-xs text-muted-foreground">
              Active players in FPL
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Teams</CardTitle>
            <Trophy className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.totalTeams || 0}</div>
            <p className="text-xs text-muted-foreground">
              Premier League teams
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average Points</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.averagePoints?.toFixed(1) || 0}</div>
            <p className="text-xs text-muted-foreground">
              Points per player
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Current Gameweek</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats?.currentGameweek || 0}</div>
            <p className="text-xs text-muted-foreground">
              Active gameweek
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Top Players */}
      <Card>
        <CardHeader>
          <CardTitle>Top Performing Players</CardTitle>
          <CardDescription>
            Players with the highest total points this season
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {topPlayers?.map((player, index) => (
              <div key={player.id} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                    <span className="text-sm font-medium text-blue-600">#{index + 1}</span>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-gray-900">{player.name}</p>
                    <p className="text-sm text-gray-500">{player.team} • {player.position}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="text-lg font-semibold text-gray-900">{player.total_points}</p>
                  <p className="text-sm text-gray-500">points</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>
            Latest updates and insights from the FPL data
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-gray-500">
            <p>Recent activity and insights will be displayed here</p>
            <p className="text-sm mt-2">Connect to the backend API to see live data</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default Dashboard;