import type { FC } from 'react';

interface LoadingSpinnerProps {
  message?: string;
  fullHeight?: boolean;
}

const LoadingSpinner: FC<LoadingSpinnerProps> = ({ message, fullHeight = false }) => {
  return (
    <div
      className={`flex flex-col items-center justify-center ${fullHeight ? 'h-64' : 'py-12'} text-gray-500`}
    >
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      {message && <p className="mt-4 text-sm">{message}</p>}
    </div>
  );
};

export default LoadingSpinner;
