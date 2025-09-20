import type { FC } from 'react';
import { ResponsiveContainer, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line } from 'recharts';
import type { TimeSeriesData } from '../../types/fpl';

interface PlayerPerformanceChartProps {
  data: TimeSeriesData[];
}

const PlayerPerformanceChart: FC<PlayerPerformanceChartProps> = ({ data }) => {
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={data}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis dataKey="gameweek" tickLine={false} axisLine={{ stroke: '#e5e7eb' }} />
        <YAxis
          yAxisId="left"
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          label={{ value: 'Points', angle: -90, position: 'insideLeft', offset: 10 }}
        />
        <YAxis
          yAxisId="right"
          orientation="right"
          tickLine={false}
          axisLine={{ stroke: '#e5e7eb' }}
          label={{ value: 'Elo Rating', angle: 90, position: 'insideRight', offset: 10 }}
        />
        <Tooltip cursor={{ stroke: '#94a3b8', strokeDasharray: '3 3' }} />
        <Legend />
        <Line
          yAxisId="left"
          type="monotone"
          dataKey="points"
          stroke="#2563eb"
          strokeWidth={2}
          dot={false}
          name="Points"
        />
        <Line
          yAxisId="right"
          type="monotone"
          dataKey="elo"
          stroke="#7c3aed"
          strokeWidth={2}
          dot={false}
          name="Elo Rating"
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

export default PlayerPerformanceChart;
