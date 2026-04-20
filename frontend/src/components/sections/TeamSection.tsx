import React, { useEffect, useRef, useState } from 'react';
import clsx from 'clsx';

const TEAM = [
  {
    name: 'Sriranjan',
    role: 'CEO & Business Development',
    description: 'Drives partnerships with SEBI-registered RIAs and PMS firms. Responsible for go-to-market strategy and advisor relationships.',
    tags: ['Strategy', 'Partnerships', 'GTM'],
    avatar: 'S',
    color: 'from-ark-red to-red-400',
  },
  {
    name: 'Mithun',
    role: 'CTO & ASRE Engine',
    description: 'Architect of the ASRE scoring engine. Built the F/T/M framework, walk-forward validation pipeline, and hash-chain infrastructure.',
    tags: ['Engineering', 'ML/Quant', 'Infrastructure'],
    avatar: 'M',
    color: 'from-blue-600 to-blue-400',
  },
  {
    name: 'Shachin',
    role: 'Research & Compliance',
    description: 'NISM Series XV certified. Ensures all ASRE outputs meet SEBI regulatory requirements. Leads research methodology and compliance frameworks.',
    tags: ['NISM XV', 'Compliance', 'Research'],
    avatar: 'S',
    color: 'from-emerald-600 to-emerald-400',
  },
];

export const TeamSection: React.FC = () => {
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
    <section ref={sectionRef} id="team" className="section-padding relative">
      <div className="section-container">
        {/* Section header */}
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1.5 text-xs font-mono text-gray-400 border border-white/10 rounded-full mb-4 uppercase tracking-wider">
            Team
          </span>
          <h2 className="section-title text-white mb-4">
            Built by <span className="gradient-text">Practitioners</span>
          </h2>
          <p className="section-subtitle mx-auto">
            A team that understands both quantitative finance and SEBI compliance — because we've lived it.
          </p>
        </div>

        {/* Team cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8 max-w-4xl mx-auto">
          {TEAM.map((member, i) => (
            <div
              key={member.name}
              className={clsx(
                'glass-card-hover p-6 lg:p-8 text-center group',
                'opacity-0',
                visible && 'animate-slide-up',
              )}
              style={{ animationDelay: `${i * 0.15}s`, animationFillMode: 'forwards' }}
            >
              {/* Avatar */}
              <div className={clsx(
                'w-16 h-16 rounded-2xl mx-auto mb-5',
                'bg-gradient-to-br flex items-center justify-center',
                'shadow-lg group-hover:scale-110 transition-transform duration-300',
                member.color,
              )}>
                <span className="text-2xl font-bold text-white">{member.avatar}</span>
              </div>

              {/* Info */}
              <h3 className="text-lg font-bold text-white mb-1">{member.name}</h3>
              <p className="text-sm text-gray-400 mb-4">{member.role}</p>
              <p className="text-xs text-gray-500 leading-relaxed mb-5">{member.description}</p>

              {/* Tags */}
              <div className="flex items-center justify-center gap-2 flex-wrap">
                {member.tags.map((tag) => (
                  <span key={tag} className="px-2.5 py-0.5 text-[10px] font-mono text-gray-500 border border-white/[0.06] rounded-full bg-white/[0.02]">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};
