'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  SparklesIcon,
  ArrowPathIcon,
  PauseIcon,
  PlayIcon,
  ClipboardDocumentCheckIcon,
  ClipboardDocumentIcon,
  BoltIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';

export interface Quote {
  id: number;
  text: string;
  author: string;
  category: 'Action' | 'Discipline' | 'Focus' | 'Psychology' | 'Momentum' | 'Grit';
}

const ANTI_PROCRASTINATION_QUOTES: Quote[] = [
  {
    id: 1,
    text: "You don't have to be great to start, but you have to start to be great.",
    author: "Zig Ziglar",
    category: "Action",
  },
  {
    id: 2,
    text: "Procrastination is the thief of time. Collar him.",
    author: "Charles Dickens",
    category: "Action",
  },
  {
    id: 3,
    text: "Amateurs sit and wait for inspiration, the rest of us just get up and go to work.",
    author: "Stephen King",
    category: "Discipline",
  },
  {
    id: 4,
    text: "The secret of getting ahead is getting started. The secret of getting started is breaking your complex overwhelming tasks into small manageable tasks, and starting on the first one.",
    author: "Mark Twain",
    category: "Momentum",
  },
  {
    id: 5,
    text: "You may delay, but time will not.",
    author: "Benjamin Franklin",
    category: "Focus",
  },
  {
    id: 6,
    text: "Action is the foundational key to all success.",
    author: "Pablo Picasso",
    category: "Action",
  },
  {
    id: 7,
    text: "In trading as in life, discipline is not a restriction; it is the ultimate vehicle for total freedom.",
    author: "Mark Douglas",
    category: "Psychology",
  },
  {
    id: 8,
    text: "Don't focus on making money; focus on protecting what you have and executing the process with ruthless discipline.",
    author: "Paul Tudor Jones",
    category: "Psychology",
  },
  {
    id: 9,
    text: "We must all suffer one of two things: the pain of discipline or the pain of regret.",
    author: "Jim Rohn",
    category: "Discipline",
  },
  {
    id: 10,
    text: "The best time to plant a tree was 20 years ago. The second best time is now.",
    author: "Chinese Proverb",
    category: "Momentum",
  },
  {
    id: 11,
    text: "A year from now you may wish you had started today.",
    author: "Karen Lamb",
    category: "Action",
  },
  {
    id: 12,
    text: "It is not because things are difficult that we do not dare; it is because we do not dare that they are difficult.",
    author: "Seneca",
    category: "Grit",
  },
  {
    id: 13,
    text: "Discipline equals freedom.",
    author: "Jocko Willink",
    category: "Discipline",
  },
  {
    id: 14,
    text: "The hard work puts you where the good luck can find you.",
    author: "Dustin Pedroia",
    category: "Grit",
  },
  {
    id: 15,
    text: "Do something today that your future self will thank you for.",
    author: "Sean Patrick Flanery",
    category: "Action",
  },
  {
    id: 16,
    text: "Small daily improvements over time lead to stunning results.",
    author: "Robin Sharma",
    category: "Momentum",
  },
  {
    id: 17,
    text: "You don't rise to the level of your goals. You fall to the level of your systems.",
    author: "James Clear",
    category: "Discipline",
  },
  {
    id: 18,
    text: "The price of inaction is far greater than the cost of making a mistake.",
    author: "Meister Eckhart",
    category: "Action",
  },
  {
    id: 19,
    text: "Stop waiting for the perfect moment. Take the moment and make it perfect through execution.",
    author: "Unknown",
    category: "Momentum",
  },
  {
    id: 20,
    text: "Losers visualize the penalties of failure. Winners visualize the rewards of relentless execution.",
    author: "William S. Burroughs",
    category: "Psychology",
  },
  {
    id: 21,
    text: "At the end of the day, you either have results or you have excuses. Choose results.",
    author: "Arnold Schwarzenegger",
    category: "Grit",
  },
  {
    id: 22,
    text: "Focus is a muscle. The more you say no to trivial distractions, the more power you bring to the essential task.",
    author: "Naval Ravikant",
    category: "Focus",
  },
  {
    id: 23,
    text: "Great acts are made up of small deeds. Start with 5 minutes of focused effort.",
    author: "Lao Tzu",
    category: "Momentum",
  },
  {
    id: 24,
    text: "Success is nothing more than a few simple disciplines, practiced every single day.",
    author: "Jim Rohn",
    category: "Discipline",
  },
  {
    id: 25,
    text: "The tragedy in life doesn't lie in not reaching your goal. The tragedy lies in having no goal to reach.",
    author: "Benjamin E. Mays",
    category: "Focus",
  },
  {
    id: 26,
    text: "If you spend too much time thinking about a thing, you'll never get it done.",
    author: "Bruce Lee",
    category: "Action",
  },
  {
    id: 27,
    text: "The successful warrior is the average person, with laser-like focus.",
    author: "Bruce Lee",
    category: "Focus",
  },
  {
    id: 28,
    text: "Never leave that till tomorrow which you can do today.",
    author: "Benjamin Franklin",
    category: "Action",
  },
  {
    id: 29,
    text: "Courage is resistance to fear, mastery of fear—not absence of fear. Take action anyway.",
    author: "Mark Twain",
    category: "Grit",
  },
  {
    id: 30,
    text: "Consistency is what transforms average into excellence.",
    author: "Tony Robbins",
    category: "Discipline",
  },
  {
    id: 31,
    text: "When you feel like quitting, remember why you started.",
    author: "Anonymous",
    category: "Grit",
  },
  {
    id: 32,
    text: "An investment in your knowledge and execution always pays the highest interest.",
    author: "Benjamin Franklin",
    category: "Psychology",
  },
  {
    id: 33,
    text: "Plan your trades. Trade your plan. Honor your stops. Cut emotional hesitation.",
    author: "Ed Seykota",
    category: "Psychology",
  },
  {
    id: 34,
    text: "Win or lose, everybody gets what they want out of the market. Some people seem to like to lose, so they win by losing money.",
    author: "Ed Seykota",
    category: "Psychology",
  },
  {
    id: 35,
    text: "The elements of good trading are: 1. Cutting losses, 2. Cutting losses, and 3. Cutting losses. If you can follow these three rules, you may have a chance.",
    author: "Ed Seykota",
    category: "Psychology",
  },
  {
    id: 36,
    text: "He who has a why to live can bear almost any how.",
    author: "Friedrich Nietzsche",
    category: "Grit",
  },
  {
    id: 37,
    text: "Do not wait; the time will never be 'just right.' Start where you stand, and work with whatever tools you may have at your command.",
    author: "George Herbert",
    category: "Action",
  },
  {
    id: 38,
    text: "Action will remove the doubt that theory cannot solve.",
    author: "Tehyi Hsieh",
    category: "Action",
  },
  {
    id: 39,
    text: "You do not need to see the whole staircase, just take the first step.",
    author: "Martin Luther King Jr.",
    category: "Momentum",
  },
  {
    id: 40,
    text: "Motivation gets you going, but discipline keeps you growing.",
    author: "John C. Maxwell",
    category: "Discipline",
  },
  {
    id: 41,
    text: "Opportunity is missed by most people because it is dressed in overalls and looks like work.",
    author: "Thomas Edison",
    category: "Grit",
  },
  {
    id: 42,
    text: "Stop overthinking. Start doing. Action creates clarity.",
    author: "Mel Robbins",
    category: "Action",
  },
  {
    id: 43,
    text: "If you want to make an easy job seem mighty hard, just keep putting off doing it.",
    author: "Olin Miller",
    category: "Momentum",
  },
  {
    id: 44,
    text: "Someday is not a day of the week.",
    author: "Denise Brennan-Nelson",
    category: "Action",
  },
  {
    id: 45,
    text: "Waste no more time arguing what a good person should be. Be one.",
    author: "Marcus Aurelius",
    category: "Discipline",
  },
  {
    id: 46,
    text: "You have power over your mind - not outside events. Realize this, and you will find immense strength.",
    author: "Marcus Aurelius",
    category: "Psychology",
  },
  {
    id: 47,
    text: "The impediment to action advances action. What stands in the way becomes the way.",
    author: "Marcus Aurelius",
    category: "Grit",
  },
  {
    id: 48,
    text: "First say to yourself what you would be; and then do what you have to do.",
    author: "Epictetus",
    category: "Focus",
  },
  {
    id: 49,
    text: "Don't explain your philosophy. Embody it.",
    author: "Epictetus",
    category: "Action",
  },
  {
    id: 50,
    text: "How long are you going to wait before you demand the best for yourself?",
    author: "Epictetus",
    category: "Momentum",
  },
  {
    id: 51,
    text: "It does not matter how slowly you go as long as you do not stop.",
    author: "Confucius",
    category: "Grit",
  },
  {
    id: 52,
    text: "The future depends on what you do today.",
    author: "Mahatma Gandhi",
    category: "Action",
  },
  {
    id: 53,
    text: "Continuous effort—not strength or intelligence—is the key to unlocking our potential.",
    author: "Winston Churchill",
    category: "Grit",
  },
  {
    id: 54,
    text: "If you're going through hell, keep going.",
    author: "Winston Churchill",
    category: "Grit",
  },
  {
    id: 55,
    text: "Yesterday you said tomorrow. Just do it.",
    author: "Nike",
    category: "Action",
  },
  {
    id: 56,
    text: "Energy flows where attention goes. Direct your focus entirely onto high-expectancy setups.",
    author: "Tony Robbins",
    category: "Focus",
  },
  {
    id: 57,
    text: "Don't count the days, make the days count.",
    author: "Muhammad Ali",
    category: "Momentum",
  },
  {
    id: 58,
    text: "I hated every minute of training, but I said, 'Don't quit. Suffer now and live the rest of your life as a champion.'",
    author: "Muhammad Ali",
    category: "Grit",
  },
  {
    id: 59,
    text: "Perfectionism is just procrastination in high-heeled shoes. Ship the work.",
    author: "Elizabeth Gilbert",
    category: "Action",
  },
  {
    id: 60,
    text: "Your mind must be stronger than your feelings. Execute what needs to be done regardless of mood.",
    author: "David Goggins",
    category: "Discipline",
  },
  {
    id: 61,
    text: "The only limit to our realization of tomorrow will be our doubts of today.",
    author: "Franklin D. Roosevelt",
    category: "Psychology",
  },
  {
    id: 62,
    text: "The man who moves a mountain begins by carrying away small stones.",
    author: "Confucius",
    category: "Momentum",
  },
  {
    id: 63,
    text: "Be tolerant with others and strict with yourself.",
    author: "Marcus Aurelius",
    category: "Discipline",
  },
  {
    id: 64,
    text: "Risk comes from not knowing what you're doing. Master the algorithm, follow the rules, eliminate hesitation.",
    author: "Warren Buffett",
    category: "Psychology",
  },
  {
    id: 65,
    text: "The difference between ordinary and extraordinary is that little extra effort applied consistently.",
    author: "Jimmy Johnson",
    category: "Momentum",
  },
];

const ROTATION_INTERVAL_SEC = 16;

export default function MotivationalQuotes() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [quotes, setQuotes] = useState<Quote[]>(ANTI_PROCRASTINATION_QUOTES);
  const [isPaused, setIsPaused] = useState(false);
  const [progress, setProgress] = useState(0);
  const [copied, setCopied] = useState(false);
  const [isFading, setIsFading] = useState(false);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize with a random quote
  useEffect(() => {
    const randomIndex = Math.floor(Math.random() * ANTI_PROCRASTINATION_QUOTES.length);
    setCurrentIndex(randomIndex);
  }, []);

  const changeQuote = useCallback((nextIdx: number) => {
    setIsFading(true);
    setTimeout(() => {
      setCurrentIndex((nextIdx + quotes.length) % quotes.length);
      setProgress(0);
      setIsFading(false);
    }, 200);
  }, [quotes.length]);

  const handleNext = useCallback(() => {
    changeQuote(currentIndex + 1);
  }, [changeQuote, currentIndex]);

  const handlePrev = useCallback(() => {
    changeQuote(currentIndex - 1);
  }, [changeQuote, currentIndex]);

  const handleShuffle = useCallback(() => {
    let nextIdx = Math.floor(Math.random() * quotes.length);
    if (nextIdx === currentIndex) {
      nextIdx = (currentIndex + 1) % quotes.length;
    }
    changeQuote(nextIdx);
  }, [changeQuote, currentIndex, quotes.length]);

  // Rotation timer and progress bar
  useEffect(() => {
    if (isPaused) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    const tickMs = 100;
    const totalTicks = (ROTATION_INTERVAL_SEC * 1000) / tickMs;

    timerRef.current = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          handleNext();
          return 0;
        }
        return prev + 100 / totalTicks;
      });
    }, tickMs);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isPaused, handleNext]);

  const currentQuote = quotes[currentIndex] || ANTI_PROCRASTINATION_QUOTES[0];

  const handleCopy = () => {
    const textToCopy = `"${currentQuote.text}" — ${currentQuote.author}`;
    navigator.clipboard.writeText(textToCopy).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const getCategoryColor = (cat: Quote['category']) => {
    switch (cat) {
      case 'Action':
        return 'bg-emerald-100 text-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-300 border-emerald-300 dark:border-emerald-700/50';
      case 'Discipline':
        return 'bg-blue-100 text-blue-800 dark:bg-blue-950/60 dark:text-blue-300 border-blue-300 dark:border-blue-700/50';
      case 'Focus':
        return 'bg-indigo-100 text-indigo-800 dark:bg-indigo-950/60 dark:text-indigo-300 border-indigo-300 dark:border-indigo-700/50';
      case 'Grit':
        return 'bg-amber-100 text-amber-800 dark:bg-amber-950/60 dark:text-amber-300 border-amber-300 dark:border-amber-700/50';
      case 'Psychology':
        return 'bg-purple-100 text-purple-800 dark:bg-purple-950/60 dark:text-purple-300 border-purple-300 dark:border-purple-700/50';
      case 'Momentum':
      default:
        return 'bg-rose-100 text-rose-800 dark:bg-rose-950/60 dark:text-rose-300 border-rose-300 dark:border-rose-700/50';
    }
  };

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-50 via-indigo-50/40 to-purple-50 dark:from-gray-800/90 dark:via-gray-800/70 dark:to-indigo-950/30 border border-indigo-100 dark:border-gray-700/70 shadow-sm transition-all duration-300 hover:shadow-md">
      {/* Progress bar line along the top */}
      <div className="w-full bg-gray-200/50 dark:bg-gray-700/40 h-1">
        <div
          className="bg-indigo-500 dark:bg-indigo-400 h-1 transition-all duration-100 ease-linear rounded-r-full"
          style={{ width: `${progress}%` }}
        />
      </div>

      <div className="p-4 sm:p-5">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          {/* Badge & Mood Pill */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-semibold rounded-full bg-indigo-600 text-white shadow-xs">
              <BoltIcon className="h-3.5 w-3.5 animate-pulse" />
              Focus & Anti-Procrastination
            </span>
            <span
              className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded-full border ${getCategoryColor(
                currentQuote.category
              )}`}
            >
              {currentQuote.category}
            </span>
            <span className="text-xs text-gray-500 dark:text-gray-400 hidden sm:inline">
              Quote {currentIndex + 1} of {quotes.length}
            </span>
          </div>

          {/* Interactive Controls */}
          <div className="flex items-center gap-1.5 self-end sm:self-auto">
            {/* Prev Quote */}
            <button
              onClick={handlePrev}
              title="Previous Quote"
              className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-white/80 dark:hover:bg-gray-700/70 transition"
            >
              <ChevronLeftIcon className="h-4 w-4" />
            </button>

            {/* Pause / Resume */}
            <button
              onClick={() => setIsPaused(!isPaused)}
              title={isPaused ? 'Resume auto-refresh' : 'Pause auto-refresh'}
              className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-white/80 dark:hover:bg-gray-700/70 transition"
            >
              {isPaused ? (
                <PlayIcon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <PauseIcon className="h-4 w-4" />
              )}
            </button>

            {/* Next / Shuffle */}
            <button
              onClick={handleShuffle}
              title="Shuffle Random Quote"
              className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium rounded-lg text-indigo-700 dark:text-indigo-300 bg-white/90 dark:bg-gray-700/80 hover:bg-white dark:hover:bg-gray-700 border border-indigo-200 dark:border-gray-600 shadow-2xs transition"
            >
              <ArrowPathIcon className="h-3.5 w-3.5" />
              <span>Next</span>
            </button>

            {/* Next Chevron */}
            <button
              onClick={handleNext}
              title="Next Quote"
              className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-white/80 dark:hover:bg-gray-700/70 transition"
            >
              <ChevronRightIcon className="h-4 w-4" />
            </button>

            {/* Copy Quote */}
            <button
              onClick={handleCopy}
              title={copied ? 'Copied to clipboard!' : 'Copy Quote'}
              className="p-1.5 rounded-lg text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-white/80 dark:hover:bg-gray-700/70 transition"
            >
              {copied ? (
                <ClipboardDocumentCheckIcon className="h-4 w-4 text-emerald-600 dark:text-emerald-400" />
              ) : (
                <ClipboardDocumentIcon className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>

        {/* Quote Content with Transition */}
        <div
          className={`mt-3.5 flex items-start gap-3 transition-opacity duration-200 ${
            isFading ? 'opacity-0' : 'opacity-100'
          }`}
        >
          <div className="hidden sm:flex shrink-0 p-2 rounded-xl bg-indigo-100/70 dark:bg-indigo-900/40 text-indigo-600 dark:text-indigo-400">
            <SparklesIcon className="h-5 w-5" />
          </div>

          <div className="flex-1">
            <blockquote className="text-base sm:text-lg font-medium text-gray-800 dark:text-gray-100 leading-snug tracking-tight">
              &ldquo;{currentQuote.text}&rdquo;
            </blockquote>
            <p className="mt-1 text-xs sm:text-sm font-semibold text-indigo-700 dark:text-indigo-300">
              — {currentQuote.author}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

