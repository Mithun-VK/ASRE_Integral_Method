import type { Variants } from 'framer-motion';

/* ═══════════════════════════════════════════════════
   Central animation variant library.
   Import named variants — never inline complex objects.
   ═══════════════════════════════════════════════════ */

// Shared easing curve (Apple-style ease-out)
const EASE_OUT: [number, number, number, number] = [0.16, 1, 0.3, 1];
const EASE_IN: [number, number, number, number] = [0.4, 0, 1, 1];

/* ─── Fade + Translate Up ─── */
export const fadeUpVariants: Variants = {
  hidden: { opacity: 0, y: 30, filter: 'blur(4px)' },
  visible: {
    opacity: 1,
    y: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.6, ease: EASE_OUT },
  },
};

/* ─── Fade + Translate from Left ─── */
export const fadeLeftVariants: Variants = {
  hidden: { opacity: 0, x: -30, filter: 'blur(4px)' },
  visible: {
    opacity: 1,
    x: 0,
    filter: 'blur(0px)',
    transition: { duration: 0.6, ease: EASE_OUT },
  },
};

/* ─── Stagger Containers ─── */
export const staggerContainerVariants: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.08, delayChildren: 0.1 },
  },
};

export const staggerContainerMobileVariants: Variants = {
  hidden: {},
  visible: {
    transition: { staggerChildren: 0.04, delayChildren: 0.05 },
  },
};

/* ─── Clip-Path Reveal (bottom → top) ─── */
export const clipRevealVariants: Variants = {
  hidden: { clipPath: 'inset(100% 0 0 0)' },
  visible: {
    clipPath: 'inset(0% 0 0 0)',
    transition: { duration: 0.7, ease: EASE_OUT },
  },
};

/* ─── Scale Pop (spring) ─── */
export const scalePopVariants: Variants = {
  hidden: { scale: 0.8, opacity: 0 },
  visible: {
    scale: 1,
    opacity: 1,
    transition: { type: 'spring', stiffness: 400, damping: 20 },
  },
};

/* ─── Card Hover ─── */
export const cardHoverVariants: Variants = {
  rest: {
    y: 0,
    boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
    transition: { duration: 0.3, ease: EASE_OUT },
  },
  hover: {
    y: -6,
    boxShadow: '0 20px 60px rgba(220,38,38,0.12)',
    transition: { duration: 0.3, ease: EASE_OUT },
  },
};

/* ─── Modal ─── */
export const modalVariants: Variants = {
  hidden: { opacity: 0, scale: 0.95, y: 20 },
  visible: {
    opacity: 1,
    scale: 1,
    y: 0,
    transition: { duration: 0.3, ease: EASE_OUT },
  },
  exit: {
    opacity: 0,
    scale: 1.02,
    y: -10,
    transition: { duration: 0.2, ease: EASE_IN },
  },
};

/* ─── Modal Backdrop ─── */
export const backdropVariants: Variants = {
  hidden: { opacity: 0 },
  visible: { opacity: 1, transition: { duration: 0.3 } },
  exit: { opacity: 0, transition: { duration: 0.2 } },
};

/* ─── Navbar scroll morph ─── */
export const navbarScrollVariants: Variants = {
  top: {
    backgroundColor: 'rgba(15,15,15,0)',
    backdropFilter: 'blur(0px)',
    borderBottomColor: 'rgba(255,255,255,0)',
    boxShadow: '0 0 0 rgba(0,0,0,0)',
  },
  scrolled: {
    backgroundColor: 'rgba(15,15,15,0.85)',
    backdropFilter: 'blur(16px)',
    borderBottomColor: 'rgba(255,255,255,0.06)',
    boxShadow: '0 4px 30px rgba(0,0,0,0.3)',
  },
};

/* ─── Accordion (FAQ) ─── */
export const accordionVariants: Variants = {
  collapsed: {
    height: 0,
    opacity: 0,
    transition: { duration: 0.3, ease: EASE_OUT },
  },
  expanded: {
    height: 'auto',
    opacity: 1,
    transition: { duration: 0.35, ease: EASE_OUT },
  },
};

/* ─── Hamburger lines → X morph ─── */
export const hamburgerTopVariants: Variants = {
  closed: { rotate: 0, y: 0 },
  open: { rotate: 45, y: 8 },
};

export const hamburgerMiddleVariants: Variants = {
  closed: { opacity: 1 },
  open: { opacity: 0 },
};

export const hamburgerBottomVariants: Variants = {
  closed: { rotate: 0, y: 0 },
  open: { rotate: -45, y: -8 },
};

/* ─── Mobile menu slide ─── */
export const mobileMenuVariants: Variants = {
  closed: {
    height: 0,
    opacity: 0,
    transition: { duration: 0.3, ease: EASE_OUT },
  },
  open: {
    height: 'auto',
    opacity: 1,
    transition: { duration: 0.35, ease: EASE_OUT, staggerChildren: 0.04 },
  },
};

export const mobileMenuItemVariants: Variants = {
  closed: { opacity: 0, x: -12 },
  open: { opacity: 1, x: 0, transition: { duration: 0.25, ease: EASE_OUT } },
};

/* ─── Slide in from right (badges) ─── */
export const slideInRightVariants: Variants = {
  hidden: { opacity: 0, x: 20 },
  visible: {
    opacity: 1,
    x: 0,
    transition: { type: 'spring', stiffness: 300, damping: 24 },
  },
};
