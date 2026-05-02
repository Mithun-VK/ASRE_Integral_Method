import React from 'react';
import { motion } from 'framer-motion';
import { Navbar } from '../components/layout/Navbar';
import { Footer } from '../components/layout/Footer';
import { HeroSection } from '../components/sections/HeroSection';
import { ProblemSection } from '../components/sections/ProblemSection';
import { ASREEngine } from '../components/sections/ASREEngine';
import { DemoWidget } from '../components/sections/DemoWidget';
import { ProductSection } from '../components/sections/ProductSection';
import { ComplianceMoat } from '../components/sections/ComplianceMoat';
import { HowItWorks } from '../components/sections/HowItWorks';
import { FAQSection } from '../components/sections/FAQSection';

// Subtle gradient divider between sections
const SectionDivider: React.FC<{ flip?: boolean }> = ({ flip }) => (
  <div
    className={`w-full h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent ${flip ? 'scale-x-[-1]' : ''}`}
    aria-hidden="true"
  />
);

const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen bg-ark-bg-primary text-gray-200 noise-bg">

      {/* Sticky scroll-progress bar */}
      <motion.div
        className="fixed top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-ark-red via-red-400 to-ark-gold origin-left z-[100] pointer-events-none"
        style={{
          scaleX: 0,
          // driven by JS scroll since CSS scroll-driven needs Safari polyfill
        }}
        id="scroll-progress"
      />

      <Navbar />

      <main id="main-content">
        {/* Hero — above the fold, no divider before */}
        <HeroSection />

        <SectionDivider />

        {/* Build understanding early */}
        <HowItWorks />

        <SectionDivider flip />

        {/* Pain points */}
        <ProblemSection />

        <SectionDivider />

        {/* Engine details */}
        <ASREEngine />

        <SectionDivider flip />

        {/* Live demo — primary conversion */}
        <DemoWidget />

        <SectionDivider />

        {/* Pricing */}
        <ProductSection />

        <SectionDivider flip />

        {/* Compliance trust */}
        <ComplianceMoat />

        <SectionDivider />

        {/* FAQ */}
        <FAQSection />
      </main>

      <Footer />
    </div>
  );
};

// Scroll progress bar wiring — runs once on mount
// Placed outside the component to avoid re-registration on re-renders
if (typeof window !== 'undefined') {
  const updateProgress = () => {
    const bar = document.getElementById('scroll-progress');
    if (!bar) return;
    const scrolled = window.scrollY;
    const total = document.documentElement.scrollHeight - window.innerHeight;
    const progress = total > 0 ? scrolled / total : 0;
    // Direct DOM manipulation — no React state, no re-render
    (bar as HTMLElement).style.transform = `scaleX(${progress})`;
  };

  window.addEventListener('scroll', updateProgress, { passive: true });
}

export default HomePage;
