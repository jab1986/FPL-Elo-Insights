import { useParams } from 'react-router-dom';
import { useTeam } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { ArrowLeft } from 'lucide-react';
import { Link } from 'react-router-dom';

const TeamDetail = () => {
  const { id } = useParams<{ id: string }>();
  const teamId = id ? parseInt(id) : 0;

  const { data: team, isLoading, error } = useTeam(teamId);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error || !team) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600">Team not found</p>
        <Link to="/teams" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
          ← Back to Teams
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center space-x-4">
        <Link to="/teams" className="text-blue-600 hover:text-blue-800">
          <ArrowLeft className="h-6 w-6" />
        </Link>
        <div>
          <h1 className="text-3xl font-bold text-gray-900">{team.name}</h1>
          <p className="text-gray-600">Position {team.position} • {team.points} points</p>
        </div>
      </div>

      {/* Team Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Position</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.position}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Points</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.points}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Games Played</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.played}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Form</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.form}</div>
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
              <p className="text-sm text-gray-600">Wins</p>
              <p className="text-lg font-semibold">{team.win}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Draws</p>
              <p className="text-lg font-semibold">{team.draw}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Losses</p>
              <p className="text-lg font-semibold">{team.loss}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Goals For</p>
              <p className="text-lg font-semibold">{team.win * 2 + team.draw}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Team Performance Chart Placeholder */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Over Time</CardTitle>
          <CardDescription>Points progression throughout the season</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-64 flex items-center justify-center text-gray-500">
            <p>Team performance chart will be displayed here</p>
            <p className="text-sm mt-2">Connect to backend API for live data</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default TeamDetail;