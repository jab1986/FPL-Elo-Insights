import type { FC } from 'react';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from 'recharts';

interface TeamPerformanceDataPoint {
  gameweek: number;
  points: number;
  wins: number;
  draws: number;
  losses: number;
}

interface TeamPerformanceChartProps {
  data: TeamPerformanceDataPoint[];
}

const TeamPerformanceChart: FC<TeamPerformanceChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="gameweek" tickLine={false} axisLine={{ stroke: '#e5e7eb' }} />
        <YAxis tickLine={false} axisLine={{ stroke: '#e5e7eb' }} />
        <Tooltip cursor={{ stroke: '#94a3b8', strokeDasharray: '3 3' }} />
        <Legend />
        <Line type="monotone" dataKey="points" stroke="#2563eb" strokeWidth={2} dot={false} name="Points" />
        <Line type="monotone" dataKey="wins" stroke="#16a34a" strokeWidth={2} dot={false} name="Wins" />
        <Line type="monotone" dataKey="draws" stroke="#facc15" strokeWidth={2} dot={false} name="Draws" />
        <Line type="monotone" dataKey="losses" stroke="#ef4444" strokeWidth={2} dot={false} name="Losses" />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default TeamPerformanceChart;
