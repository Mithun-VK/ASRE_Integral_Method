'use client'; // remove if Vite, keep if Next.js App Router

import React, { useState, useEffect, useRef } from 'react';
import { PopupModal, useCalendlyEventListener } from 'react-calendly';
import { motion, AnimatePresence } from 'framer-motion';
import { CTAButton } from '../ui/CTAButton';

// ── Replace with your actual Calendly URL ────────────────────────────────────
const CALENDLY_URL = 'https://calendly.com/mithunvk216/30min';

const SPRING = [0.16, 1, 0.3, 1] as const;

interface BookDemoProps {
    variant?: 'primary' | 'secondary';
    label?: string;
    className?: string;
    id?: string;
    size?: 'sm' | 'md' | 'lg';
}

export const BookDemo: React.FC<BookDemoProps> = ({
    variant = 'primary',
    label = 'Book a Demo',
    className = '',
    id = 'book-demo',
    size = 'lg',
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const [booked, setBooked] = useState(false);
    const rootRef = useRef<HTMLElement | null>(null);

    useEffect(() => {
        rootRef.current = document.getElementById('root') ?? document.body;
    }, []);

    useCalendlyEventListener({
        onEventScheduled: () => {
            setBooked(true);
            setIsOpen(false);

            const timer = window.setTimeout(() => {
                setBooked(false);
            }, 5000);

            return () => window.clearTimeout(timer);
        },
    });

    return (
        <>
            {/* ── CTAButton trigger ── */}
            <div className={className}>
                <CTAButton
                    variant={variant}
                    size={size}
                    id={id}
                    onClick={() => setIsOpen(true)}
                    icon={
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                            />
                        </svg>
                    }
                >
                    {label}
                </CTAButton>
            </div>

            {/* ── Booking confirmed toast ── */}
            <AnimatePresence>
                {booked && (
                    <motion.div
                        className="fixed bottom-6 right-6 z-[9999] flex items-center gap-3 px-4 py-3 rounded-xl bg-emerald-950/90 border border-emerald-500/30 backdrop-blur-md shadow-xl"
                        initial={{ opacity: 0, y: 24, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 12, scale: 0.95 }}
                        transition={{ duration: 0.4, ease: SPRING }}
                        role="status"
                        aria-live="polite"
                    >
                        <span className="text-emerald-400 shrink-0">
                            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M5 13l4 4L19 7"
                                />
                            </svg>
                        </span>

                        <div>
                            <p className="text-sm font-semibold text-white">Demo booked!</p>
                            <p className="text-xs text-gray-400">
                                Check your inbox for the confirmation.
                            </p>
                        </div>

                        <button
                            className="ml-2 text-gray-500 hover:text-gray-300 transition-colors"
                            onClick={() => setBooked(false)}
                            aria-label="Dismiss"
                        >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M6 18L18 6M6 6l12 12"
                                />
                            </svg>
                        </button>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Calendly popup modal ── */}
            {rootRef.current && (
                <PopupModal
                    url={CALENDLY_URL}
                    open={isOpen}
                    onModalClose={() => setIsOpen(false)}
                    rootElement={rootRef.current}
                    pageSettings={{
                        backgroundColor: '060608',
                        primaryColor: 'DC2626',
                        textColor: 'e8e6ff',
                        hideLandingPageDetails: false,
                        hideEventTypeDetails: false,
                    }}
                    prefill={{
                        // email: user?.email,
                        // name: user?.name,
                    }}
                    utm={{
                        utmSource: 'ark-angl-website',
                        utmMedium: 'book-demo-button',
                        utmCampaign: 'hero-cta',
                    }}
                />
            )}
        </>
    );
};
