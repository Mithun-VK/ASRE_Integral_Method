import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

const PROBLEMS = [
  {
    quote: '"I use broker notes"',
    problem: 'No Run ID / Hash. SEBI audits BASIS.',
    description: 'Broker notes lack traceable audit trails. When SEBI requests documentation of your investment rationale, you need timestamped, hash-signed proof — not forwarded emails.',
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
    risk: 'High audit risk',
  },
  {
    quote: '"Excel has my model"',
    problem: 'Retroactively editable. No timestamp proof.',
    description: 'Spreadsheets can be modified after the fact. There\'s no immutable record proving your analysis existed before the investment decision was made.',
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
      </svg>
    ),
    risk: 'Compliance gap',
  },
  {
    quote: '"My own analysis"',
    problem: 'Can you document F/T/M scores in 5 min?',
    description: 'Manual analysis can\'t produce structured Fundamental, Technical, and Momentum scores with hash proofs fast enough to scale across your client portfolio.',
    icon: (
      <svg className="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    risk: 'Scalability limit',
  },
];

export const ProblemSection: React.FC = () => {
  const sectionRef = useRef<HTMLElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) setVisible(true); },
      { threshold: 0.15 }
    );
    if (sectionRef.current) observer.observe(sectionRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <section ref={sectionRef} id="problem" className="section-padding relative">
      <div className="section-container">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-ark-red border border-ark-red/20 rounded-full mb-4 uppercase tracking-wider">
            The Problem
          </span>
          <h2 className="section-title text-white mb-4">
            What RIAs Tell Us <span className="gradient-text">Every Week</span>
          </h2>
          <p className="section-subtitle mx-auto">
            Three conversations we hear from every SEBI-registered advisor before they switch to ASRE.
          </p>
        </div>

        {/* Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {PROBLEMS.map((item, i) => (
            <div
              key={i}
              className={clsx(
                'glass-card-hover p-6 lg:p-8 group',
                'opacity-0',
                visible && 'animate-slide-up',
              )}
              style={{ animationDelay: `${i * 0.15}s`, animationFillMode: 'forwards' }}
            >
              {/* Icon */}
              <div className="w-12 h-12 rounded-xl bg-ark-red/10 border border-ark-red/20 flex items-center justify-center text-ark-red mb-5 group-hover:bg-ark-red/20 transition-colors">
                {item.icon}
              </div>

              {/* Quote */}
              <p className="text-xl font-bold text-white mb-2">{item.quote}</p>

              {/* Problem highlight */}
              <p className="text-sm font-mono text-ark-red mb-4">{item.problem}</p>

              {/* Description */}
              <p className="text-sm text-gray-400 leading-relaxed mb-4">{item.description}</p>

              {/* Risk tag */}
              <div className="flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-red-500" />
                <span className="text-xs text-gray-500 uppercase tracking-wider">{item.risk}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
