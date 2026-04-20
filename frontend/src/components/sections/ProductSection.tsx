import React, { useState, useEffect, useRef } from 'react';
import clsx from 'clsx';
import { PricingCard } from '../ui/PricingCard';

type Tab = 'ria' | 'pms';

const PRODUCTS = {
  ria: [
    {
      id: 'on-demand',
      title: 'On-Demand Report',
      price: '₹500',
      period: 'stock',
      description: 'Single stock ASRE score with full F/T/M breakdown, dip context, and hash-signed PDF.',
      features: [
        'F/T/M composite score',
        'Dip context analysis',
        'SHA-256 hash proof',
        'SEBI AI disclosure',
        'PDF delivery in 4 seconds',
      ],
      cta: 'Get Sample Report',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      id: 'research-pack',
      title: 'Research Pack',
      price: '₹8-10k',
      period: 'month',
      description: 'Weekly curated coverage of top-performing stocks. Perfect for client-facing research notes.',
      features: [
        'Friday stock coverage batch',
        '10-15 stocks per pack',
        'Sector-wise breakdown',
        'Client-ready formatting',
        'IA/RA mode toggle',
        'Priority email support',
      ],
      cta: 'Book Demo →',
      popular: true,
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      ),
    },
    {
      id: 'backtest',
      title: 'Backtest Report',
      price: '₹25-75k',
      period: 'report',
      description: 'Deep-dive walk-forward validation of your investment thesis or strategy across market cycles.',
      features: [
        '545+ walk-forward iterations',
        'Custom strategy parameters',
        'Drawdown analysis',
        'Risk-adjusted returns',
        'Board-ready PDF',
      ],
      cta: 'Request Quote →',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      id: 'dashboard',
      title: 'Advisor Dashboard',
      price: '₹15-25k',
      period: 'seat/month',
      description: 'Multi-advisor scoring platform with portfolio-wide ASRE analytics, bulk scoring, and API access.',
      features: [
        'Multi-seat access',
        'Portfolio-wide scoring',
        'Bulk CSV upload',
        'REST API access',
        'Custom branding',
        'Dedicated account manager',
      ],
      cta: 'Contact Sales →',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
        </svg>
      ),
    },
  ],
  pms: [
    {
      id: 'pms-on-demand',
      title: 'On-Demand Report',
      price: '₹750',
      period: 'stock',
      description: 'Institutional-grade ASRE report with enhanced compliance formatting for PMS mandates.',
      features: [
        'PMS-specific formatting',
        'F/T/M composite score',
        'Enhanced compliance notes',
        'SHA-256 hash proof',
        'SEBI PMS disclosure',
      ],
      cta: 'Get Sample Report',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      ),
    },
    {
      id: 'pms-research-pack',
      title: 'Research Pack',
      price: '₹15-20k',
      period: 'month',
      description: 'Comprehensive weekly coverage tailored for portfolio management services with sector allocation insights.',
      features: [
        'Friday stock coverage batch',
        '20-30 stocks per pack',
        'Sector allocation overlay',
        'PMS mandate alignment',
        'Board presentation format',
        'Priority support',
      ],
      cta: 'Book Demo →',
      popular: true,
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
        </svg>
      ),
    },
    {
      id: 'pms-backtest',
      title: 'Backtest Report',
      price: '₹50-100k',
      period: 'report',
      description: 'Institutional walk-forward validation with compliance appendix for PMS audit documentation.',
      features: [
        '545+ walk-forward iterations',
        'Compliance appendix',
        'Board-ready format',
        'Multi-benchmark analysis',
        'Risk decomposition',
      ],
      cta: 'Request Quote →',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      id: 'pms-dashboard',
      title: 'Advisor Dashboard',
      price: '₹30-50k',
      period: 'seat/month',
      description: 'Enterprise PMS platform with multi-portfolio scoring, white-label reports, and dedicated infrastructure.',
      features: [
        'Multi-portfolio scoring',
        'White-label PDF reports',
        'Dedicated infrastructure',
        'REST API + webhooks',
        'SOC 2 compliance ready',
        'Dedicated success manager',
      ],
      cta: 'Contact Sales →',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z" />
        </svg>
      ),
    },
  ],
};

export const ProductSection: React.FC = () => {
  const [tab, setTab] = useState<Tab>('ria');
  const sectionRef = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.1 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  const scrollToDemo = () => {
    const el = document.querySelector('#demo');
    if (el) el.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section ref={sectionRef} id="pricing" className="section-padding relative">
      <div className="section-container">
        {/* Section header */}
        <div className="text-center mb-12">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-gold border border-ark-gold/20 rounded-full mb-4 uppercase tracking-wider">
            Pricing
          </span>
          <h2 className="section-title text-white mb-4">
            Built for <span className="gradient-text-gold">Your Practice</span>
          </h2>
          <p className="section-subtitle mx-auto">
            From single-stock reports to enterprise dashboards. Choose what fits your practice.
          </p>
        </div>

        {/* RIA / PMS Toggle */}
        <div className="flex justify-center mb-12">
          <div className="inline-flex rounded-xl bg-white/[0.03] border border-white/[0.06] p-1">
            {(['ria', 'pms'] as Tab[]).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className={clsx(
                  'px-6 py-2.5 text-sm font-semibold rounded-lg transition-all duration-300',
                  tab === t
                    ? 'bg-ark-red text-white shadow-lg shadow-ark-red/25'
                    : 'text-gray-400 hover:text-white'
                )}
                id={`pricing-tab-${t}`}
              >
                {t === 'ria' ? 'RIA / IA' : 'PMS'}
              </button>
            ))}
          </div>
        </div>

        {/* Pricing cards — snap scroll on mobile, grid on md+ */}
        <div className="relative">
          {/* Mobile: horizontal snap scroll */}
          <div className={clsx(
            'flex md:hidden gap-4 overflow-x-auto snap-x snap-mandatory pb-4 -mx-4 px-4',
            'scrollbar-hide',
            'opacity-0',
            visible && 'animate-fade-in',
          )} style={{ animationFillMode: 'forwards' }}>
            {PRODUCTS[tab].map((product) => (
              <div key={product.id} className="snap-center flex-shrink-0 w-[80vw] max-w-[320px]">
                <PricingCard
                  id={product.id}
                  title={product.title}
                  price={product.price}
                  period={product.period}
                  description={product.description}
                  features={product.features}
                  cta={product.cta}
                  popular={product.popular || false}
                  icon={product.icon}
                  ctaAction={scrollToDemo}
                />
              </div>
            ))}
          </div>

          {/* Tablet/Desktop: grid */}
          <div className={clsx(
            'hidden md:grid md:grid-cols-2 xl:grid-cols-4 gap-6 lg:gap-8',
            'opacity-0',
            visible && 'animate-fade-in',
          )} style={{ animationFillMode: 'forwards' }}>
            {PRODUCTS[tab].map((product) => (
              <PricingCard
                key={product.id}
                id={product.id}
                title={product.title}
                price={product.price}
                period={product.period}
                description={product.description}
                features={product.features}
                cta={product.cta}
                popular={product.popular || false}
                icon={product.icon}
                ctaAction={scrollToDemo}
              />
            ))}
          </div>
        </div>

        {/* Bottom note */}
        <p className="text-center text-xs text-gray-600 mt-8">
          All prices exclusive of GST. Custom enterprise pricing available.{' '}
          <button onClick={scrollToDemo} className="text-ark-red hover:underline">Contact us →</button>
        </p>
      </div>
    </section>
  );
};
