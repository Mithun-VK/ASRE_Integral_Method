import { AnimatePresence, motion } from 'framer-motion';
import { ScrollProgressBar } from './components/ui/ScrollProgressBar';
import HomePage from './pages/HomePage';

function App() {
  return (
    <>
      <ScrollProgressBar />
      <AnimatePresence mode="wait">
        <motion.div
          key="home"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
        >
          <HomePage />
        </motion.div>
      </AnimatePresence>
    </>
  );
}

export default App;
