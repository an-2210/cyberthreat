import { useEffect, useState } from 'react';

// Runs a fetch once when the component mounts and exposes its data, error and loading state.
export function useLoader<T>(load: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    load().then(setData).catch((e: Error) => setError(e.message));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, error, loading: !data && !error };
}

export function ErrorBox({ message }: { message: string }) {
  return (
    <div className="notice error">
      <strong>Could not load this from the backend.</strong> {message}
      <div className="muted">Start it with <code>uvicorn app.backend.main:app --port 8000</code> from the repo root.</div>
    </div>
  );
}
