import { useMemo, useState } from 'react';
import { usePlayers } from '../hooks/useFPLData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Link } from 'react-router-dom';
import { Search, Filter, ArrowUpDown } from 'lucide-react';
import LoadingSpinner from '../components/LoadingSpinner';

const Players = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [positionFilter, setPositionFilter] = useState('');
  const [teamFilter, setTeamFilter] = useState('');
  const [sortOption, setSortOption] = useState<'points-desc' | 'price-asc' | 'price-desc' | 'name-asc'>('points-desc');

  const filters: Record<string, string> = {};
  if (positionFilter) filters.position = positionFilter;
  if (teamFilter) filters.team = teamFilter;

  const { data: players, isLoading, error } = usePlayers(filters);

  const availableTeams = useMemo(() => {
    return Array.from(new Set(players?.map((player) => player.team) ?? [])).sort();
  }, [players]);

  const filteredPlayers = useMemo(() => {
    return players?.filter((player) =>
      player.name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [players, searchTerm]);

  const sortedPlayers = useMemo(() => {
    if (!filteredPlayers) {
      return [];
    }

    const playersCopy = [...filteredPlayers];
    switch (sortOption) {
      case 'price-asc':
        return playersCopy.sort((a, b) => a.now_cost - b.now_cost);
      case 'price-desc':
        return playersCopy.sort((a, b) => b.now_cost - a.now_cost);
      case 'name-asc':
        return playersCopy.sort((a, b) => a.name.localeCompare(b.name));
      case 'points-desc':
      default:
        return playersCopy.sort((a, b) => b.total_points - a.total_points);
    }
  }, [filteredPlayers, sortOption]);

  const positions = ['Goalkeeper', 'Defender', 'Midfielder', 'Forward'];

  if (isLoading) {
    return <LoadingSpinner message="Loading player pool..." fullHeight />;
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <p className="text-red-600 font-medium">Error loading players data. Please refresh and try again.</p>
      </div>
    );
  }

  const hasResults = sortedPlayers.length > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Players</h1>
          <p className="text-gray-600 mt-1">
            Browse and analyze Fantasy Premier League players with advanced filters and sorting.
          </p>
        </div>
        <Badge variant="secondary">{sortedPlayers.length} Players</Badge>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center">
            <Filter className="h-5 w-5 mr-2" />
            Filters &amp; Sorting
          </CardTitle>
          <CardDescription>Refine the player list by position, team, and price or point trends.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {/* Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Search Players
              </label>
              <div className="relative">
                <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <input
                  type="text"
                  placeholder="Search by name..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>
            </div>

            {/* Position Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Position</label>
              <select
                value={positionFilter}
                onChange={(e) => setPositionFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Positions</option>
                {positions.map((pos) => (
                  <option key={pos} value={pos}>
                    {pos}
                  </option>
                ))}
              </select>
            </div>

            {/* Team Filter */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Team</label>
              <select
                value={teamFilter}
                onChange={(e) => setTeamFilter(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">All Teams</option>
                {availableTeams.map((team) => (
                  <option key={team} value={team}>
                    {team}
                  </option>
                ))}
              </select>
            </div>

            {/* Sort */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Sort By</label>
              <div className="relative">
                <ArrowUpDown className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                <select
                  value={sortOption}
                  onChange={(e) => setSortOption(e.target.value as typeof sortOption)}
                  className="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="points-desc">Points (High to Low)</option>
                  <option value="price-desc">Price (High to Low)</option>
                  <option value="price-asc">Price (Low to High)</option>
                  <option value="name-asc">Name (A-Z)</option>
                </select>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Players Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sortedPlayers.map((player) => (
          <Card key={player.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-center justify-between">
                <CardTitle className="text-lg">{player.name}</CardTitle>
                <Badge variant="secondary">{player.position}</Badge>
              </div>
              <CardDescription>{player.team}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Total Points:</span>
                  <span className="font-semibold">{player.total_points}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Price:</span>
                  <span className="font-semibold">£{(player.now_cost / 10).toFixed(1)}m</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Selected by:</span>
                  <span className="font-semibold">{player.selected_by_percent}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Points per Game:</span>
                  <span className="font-semibold">{player.points_per_game}</span>
                </div>
              </div>
              <div className="mt-4">
                <Link
                  to={`/players/${player.id}`}
                  className="w-full bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors text-center block"
                >
                  View Details
                </Link>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {!hasResults && (
        <div className="text-center py-12">
          <p className="text-gray-500">No players found matching your criteria. Adjust the filters to discover more options.</p>
        </div>
      )}
    </div>
  );
};

export default Players;
