import React from 'react';
import clsx from 'clsx';
import { CTAButton } from './CTAButton';

interface PricingCardProps {
  title: string;
  price: string;
  period?: string;
  description: string;
  features: string[];
  cta: string;
  ctaAction?: () => void;
  popular?: boolean;
  icon: React.ReactNode;
  id: string;
}

export const PricingCard: React.FC<PricingCardProps> = ({
  title,
  price,
  period,
  description,
  features,
  cta,
  ctaAction,
  popular = false,
  icon,
  id,
}) => {
  return (
    <div
      className={clsx(
        'relative flex flex-col rounded-2xl border p-6 lg:p-8',
        'transition-all duration-500',
        'hover:-translate-y-2 hover:shadow-2xl',
        popular
          ? 'bg-gradient-to-b from-ark-red/[0.08] to-transparent border-ark-red/30 shadow-lg shadow-ark-red/10 hover:shadow-ark-red/20'
          : 'bg-white/[0.02] border-white/[0.06] hover:border-white/[0.12] hover:bg-white/[0.04]',
      )}
      id={id}
    >
      {popular && (
        <div className="absolute -top-3 left-1/2 -translate-x-1/2">
          <span className="px-4 py-1 text-xs font-bold bg-ark-red text-white rounded-full uppercase tracking-wider shadow-lg shadow-ark-red/30">
            Most Popular
          </span>
        </div>
      )}

      <div className="flex items-center gap-3 mb-4">
        <div className={clsx(
          'w-10 h-10 rounded-xl flex items-center justify-center',
          popular ? 'bg-ark-red/15 text-ark-red' : 'bg-white/[0.06] text-gray-400',
        )}>
          {icon}
        </div>
        <h3 className="text-lg font-bold text-white">{title}</h3>
      </div>

      <div className="mb-4">
        <span className={clsx(
          'text-3xl font-bold',
          popular ? 'gradient-text' : 'text-white'
        )}>
          {price}
        </span>
        {period && <span className="text-sm text-gray-500 ml-1">/{period}</span>}
      </div>

      <p className="text-sm text-gray-400 mb-6">{description}</p>

      <ul className="space-y-3 mb-8 flex-grow">
        {features.map((feature, i) => (
          <li key={i} className="flex items-start gap-2 text-sm text-gray-300">
            <svg className={clsx('w-4 h-4 flex-shrink-0 mt-0.5', popular ? 'text-ark-red' : 'text-gray-500')} fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
            {feature}
          </li>
        ))}
      </ul>

      <CTAButton
        variant={popular ? 'primary' : 'secondary'}
        size="md"
        className="w-full"
        onClick={ctaAction}
        id={`${id}-cta`}
      >
        {cta}
      </CTAButton>
    </div>
  );
};
