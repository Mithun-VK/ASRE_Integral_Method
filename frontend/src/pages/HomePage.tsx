import React from 'react';
import { Navbar } from '../components/layout/Navbar';
import { Footer } from '../components/layout/Footer';
import { HeroSection } from '../components/sections/HeroSection';
import { ProblemSection } from '../components/sections/ProblemSection';
import { ASREEngine } from '../components/sections/ASREEngine';
import { DemoWidget } from '../components/sections/DemoWidget';
import { ProductSection } from '../components/sections/ProductSection';
import { ComplianceMoat } from '../components/sections/ComplianceMoat';
import { HowItWorks } from '../components/sections/HowItWorks';
import { TeamSection } from '../components/sections/TeamSection';
import { FAQSection } from '../components/sections/FAQSection';

const HomePage: React.FC = () => {
  return (
    <div className="min-h-screen bg-ark-bg-primary text-gray-200 noise-bg">
      <Navbar />

      {/* Hero - Above the fold */}
      <HeroSection />

      {/* How it works - to build understanding early */}
      <HowItWorks />

      {/* Pain points */}
      <ProblemSection />

      {/* Engine details */}
      <ASREEngine />

      {/* LIVE DEMO - Primary conversion engine */}
      <DemoWidget />

      {/* Pricing */}
      <ProductSection />

      {/* Compliance trust */}
      <ComplianceMoat />

      {/* Team */}
      <TeamSection />

      {/* FAQ */}
      <FAQSection />

      <Footer />
    </div>
  );
};

export default HomePage;