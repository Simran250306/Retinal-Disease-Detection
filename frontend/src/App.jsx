import { NavLink, Route, Routes } from "react-router-dom";
import Home from "./routes/Home.jsx";
import About from "./routes/About.jsx";

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <div>
          <div className="title">Retinal Disease Detection</div>
          <div className="subtitle">
            React UI connected to a .cbm model API with confidence thresholding.
          </div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Predict
          </NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </div>
  );
}