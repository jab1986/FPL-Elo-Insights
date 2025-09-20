import { useTeams } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Link } from 'react-router-dom';

const Teams = () => {
  const { data: teams, isLoading, error } = useTeams();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-600">Error loading teams data</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Teams</h1>
          <p className="text-gray-600 mt-1">
            Premier League teams and their performance
          </p>
        </div>
        <Badge variant="secondary">
          {teams?.length || 0} Teams
        </Badge>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {teams?.map((team) => (
          <Card key={team.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <CardTitle className="text-lg">{team.name}</CardTitle>
              <CardDescription>Position: {team.position}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Points:</span>
                  <span className="font-semibold">{team.points}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Played:</span>
                  <span className="font-semibold">{team.played}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Form:</span>
                  <span className="font-semibold">{team.form}</span>
                </div>
              </div>
              <div className="mt-4">
                <Link
                  to={`/teams/${team.id}`}
                  className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors text-center block"
                >
                  View Details
                </Link>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};

export default Teams;