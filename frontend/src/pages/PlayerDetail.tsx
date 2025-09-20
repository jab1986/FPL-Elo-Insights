import { useParams } from 'react-router-dom';
import { usePlayer } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const PlayerDetail = () => {
  const { id } = useParams<{ id: string }>();
  const playerId = id ? parseInt(id) : 0;

  const { data: player, isLoading, error } = usePlayer(playerId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !player) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600">Player not found</p>
        <Link to="/players" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
          ← Back to Players
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/players" className="text-blue-600 hover:text-blue-800">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{player.name}</h1>
          <p className="text-gray-600">{player.team} • {player.position}</p>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Total Points</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{player.total_points}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Price</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">£{(player.now_cost / 10).toFixed(1)}m</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Selected By</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{player.selected_by_percent}%</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Points per Game</CardTitle>
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

      {/* Performance Chart Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Over Time</CardTitle>
          <CardDescription>Points and Elo rating progression</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-gray-500">
            <p>Performance chart will be displayed here</p>
            <p className="text-sm mt-2">Connect to backend API for live data</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PlayerDetail;