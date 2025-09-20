import { useMatches, useCurrentGameweek } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';

const Matches = () => {
  const { data: currentGameweek } = useCurrentGameweek();
  const { data: matches, isLoading, error } = useMatches(currentGameweek?.gameweek);

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
        <p className="text-red-600">Error loading matches data</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Matches</h1>
          <p className="text-gray-600 mt-1">
            Current gameweek fixtures and results
          </p>
        </div>
        {currentGameweek && (
          <Badge variant="secondary">
            Gameweek {currentGameweek.gameweek}
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {matches?.map((match) => (
          <Card key={match.id}>
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">
                  {match.home_team} vs {match.away_team}
                </CardTitle>
                <Badge variant={match.finished ? "default" : "secondary"}>
                  {match.finished ? "Finished" : "Upcoming"}
                </Badge>
              </div>
              <CardDescription>
                {new Date(match.kickoff_time).toLocaleDateString()}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {match.finished ? (
                <div className="text-center">
                  <div className="text-2xl font-bold">
                    {match.home_score} - {match.away_score}
                  </div>
                </div>
              ) : (
                <div className="text-center text-gray-500">
                  <p>Match not started</p>
                  <p className="text-sm">
                    {new Date(match.kickoff_time).toLocaleTimeString()}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {matches?.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-500">No matches found for this gameweek</p>
        </div>
      )}
    </div>
  );
};

export default Matches;