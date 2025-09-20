import { Link, useParams } from 'react-router-dom';
import { useTeam, useTeamPerformanceOverTime, useMatches } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { ArrowLeft } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';
import TeamPerformanceChart from '../components/charts/TeamPerformanceChart';

const TeamDetail = () => {
  const { id } = useParams<{ id: string }>();
  const teamId = id ? Number.parseInt(id, 10) : 0;

  const {
    data: team,
    isLoading,
    error,
  } = useTeam(teamId);
  const {
    data: performance,
    isLoading: performanceLoading,
    error: performanceError,
  } = useTeamPerformanceOverTime(teamId);
  const {
    data: matches,
    isLoading: matchesLoading,
    error: matchesError,
  } = useMatches();

  if (isLoading) {
    return <LoadingSpinner message="Loading team profile..." fullHeight />;
  }

  if (error || !team) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 font-medium">Team not found.</p>
        <Link to="/teams" className="text-blue-600 hover:text-blue-800 mt-4 inline-block">
          ← Back to Teams
        </Link>
      </div>
    );
  }

  const recentMatches = matches
    ?.filter((match) => match.home_team === team.name || match.away_team === team.name)
    .sort(
      (a, b) => new Date(b.kickoff_time).getTime() - new Date(a.kickoff_time).getTime()
    )
    .slice(0, 5);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div className="flex items-center space-x-4">
          <Link to="/teams" className="text-blue-600 hover:text-blue-800">
            <ArrowLeft className="h-6 w-6" />
          </Link>
          <div>
            <h1 className="text-3xl font-bold text-gray-900">{team.name}</h1>
            <p className="text-gray-600">{team.points} points • Position {team.position}</p>
          </div>
        </div>
        <Badge variant="secondary" className="w-fit px-4 py-2 text-sm font-semibold">
          Overall strength: {team.strength}
        </Badge>
      </div>

      {/* Team Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Points</CardTitle>
            <CardDescription>Total accumulated</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.points}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Games Played</CardTitle>
            <CardDescription>Premier League fixtures</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.played}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Form</CardTitle>
            <CardDescription>Recent run</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.form || 'N/A'}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Overall Strength</CardTitle>
            <CardDescription>Home &amp; away composite</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{team.strength}</div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Stats */}
      <Card>
        <CardHeader>
          <CardTitle>Season Statistics</CardTitle>
          <CardDescription>League record and core metrics.</CardDescription>
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
              <p className="text-sm text-gray-600">Goals (approx.)</p>
              <p className="text-lg font-semibold">{team.win * 2 + team.draw}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Home Strength</p>
              <p className="text-lg font-semibold">{team.strength_overall_home}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Away Strength</p>
              <p className="text-lg font-semibold">{team.strength_overall_away}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Attack Rating</p>
              <p className="text-lg font-semibold">{team.strength_attack_home}/{team.strength_attack_away}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Defence Rating</p>
              <p className="text-lg font-semibold">{team.strength_defence_home}/{team.strength_defence_away}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Over Time</CardTitle>
          <CardDescription>Points trajectory and results balance throughout the season.</CardDescription>
        </CardHeader>
        <CardContent>
          {performanceLoading ? (
            <LoadingSpinner />
          ) : performanceError ? (
            <p className="text-sm text-red-600">Unable to load performance data.</p>
          ) : performance && performance.length > 0 ? (
            <TeamPerformanceChart data={performance} />
          ) : (
            <p className="text-sm text-gray-500">Performance data will appear once matches are available.</p>
          )}
        </CardContent>
      </Card>

      {/* Recent Matches */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Matches</CardTitle>
          <CardDescription>Latest fixtures involving {team.name}.</CardDescription>
        </CardHeader>
        <CardContent>
          {matchesLoading ? (
            <LoadingSpinner />
          ) : matchesError ? (
            <p className="text-sm text-red-600">Unable to load match data.</p>
          ) : recentMatches && recentMatches.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Fixture</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Date</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Status</th>
                    <th scope="col" className="px-4 py-3 text-left font-semibold text-gray-700">Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 bg-white">
                  {recentMatches.map((match) => {
                    const isHome = match.home_team === team.name;
                    const opponent = isHome ? match.away_team : match.home_team;
                    const homeScore = match.home_score;
                    const awayScore = match.away_score;
                    const hasScore = match.finished && homeScore !== null && awayScore !== null;
                    const score = hasScore ? `${homeScore} - ${awayScore}` : 'TBD';

                    let resultLabel: string;
                    if (!hasScore) {
                      resultLabel = 'Upcoming';
                    } else if (homeScore === awayScore) {
                      resultLabel = 'Draw';
                    } else if ((isHome && homeScore > awayScore) || (!isHome && awayScore > homeScore)) {
                      resultLabel = 'Win';
                    } else {
                      resultLabel = 'Loss';
                    }

                    return (
                      <tr key={match.id} className="hover:bg-gray-50">
                        <td className="px-4 py-3 font-medium text-gray-900">
                          {isHome ? 'vs' : 'at'} {opponent}
                        </td>
                        <td className="px-4 py-3 text-gray-700">
                          {new Date(match.kickoff_time).toLocaleDateString()} •
                          {' '}
                          {new Date(match.kickoff_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td className="px-4 py-3 text-gray-700">{resultLabel}</td>
                        <td className="px-4 py-3 text-gray-700">{score}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <p className="text-sm text-gray-500">Match data will appear once fixtures are available.</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default TeamDetail;
