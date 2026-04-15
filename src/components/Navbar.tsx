import React from 'react';
import { Cat, Info } from 'lucide-react';

const Navbar: React.FC = () => (
  <header className="nav-bar" id="navbar">
    <div className="nav-brand">
      <Cat size={24} />
      <span>Shiretto Cat</span>
    </div>
    <button className="nav-circular-btn" id="btn-info" aria-label="情報">
      <Info size={20} />
    </button>
  </header>
);

export default Navbar;
