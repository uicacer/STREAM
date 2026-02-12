/**
 * index.ts - Central Export for All Types
 * ========================================
 *
 * WHAT IS THIS FILE FOR?
 * Instead of importing from individual files:
 *   import { Message } from './types/message'
 *   import { Conversation } from './types/conversation'
 *   import { ChatSettings } from './types/settings'
 *
 * You can import everything from one place:
 *   import { Message, Conversation, ChatSettings } from './types'
 *
 * This is called a "barrel export" - it re-exports everything
 * from multiple files through a single entry point.
 *
 * BENEFITS:
 * - Cleaner imports
 * - Easier refactoring (move types between files without breaking imports)
 * - Better organized codebase
 */

export * from './message'
export * from './conversation'
export * from './settings'
