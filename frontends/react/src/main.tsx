/**
 * main.tsx - The Entry Point of Our React Application
 * ====================================================
 *
 * This file is where React "boots up" and connects to the HTML page.
 *
 * WHAT HAPPENS HERE:
 * 1. We import React and ReactDOM (React's connection to the browser)
 * 2. We import our main App component
 * 3. We import our global CSS styles
 * 4. We tell React: "Find the <div id='root'> in index.html and render our app there"
 *
 * WITHOUT THIS FILE:
 * - React wouldn't know where to render in the HTML page
 * - The browser would just show an empty <div id="root"></div>
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import './styles/globals.css'

/**
 * ReactDOM.createRoot() - Creates a "root" where React will manage the DOM
 *
 * document.getElementById('root')! - Gets the <div id="root"> from index.html
 *   The "!" tells TypeScript "I promise this element exists, trust me"
 *
 * .render() - Tells React what to display inside that root element
 */
ReactDOM.createRoot(document.getElementById('root')!).render(
  /**
   * React.StrictMode - A development tool that:
   * - Warns about deprecated features
   * - Warns about potential problems
   * - Runs components twice in dev mode to catch bugs
   *
   * It has NO effect in production builds - only helps during development.
   * You can remove it if the double-rendering confuses you during debugging.
   *
   * ErrorBoundary - Catches React errors and shows a fallback UI
   * instead of crashing the entire app and showing a blank page.
   */
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
)
