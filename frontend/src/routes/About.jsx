export default function About() {
  return (
    <section className="card">
      <h3>About This Demo</h3>
      <div className="about">
        <p>
          This UI is a Vite + React app that sends an image to the FastAPI backend
          and displays class probabilities from the CatBoost .cbm model.
        </p>
        <p>
          Edit the API base URL on the Predict screen if your backend runs on a
          different host or port.
        </p>
      </div>
    </section>
  );
}