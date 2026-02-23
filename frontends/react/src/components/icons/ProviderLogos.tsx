/**
 * ProviderLogos.tsx - AI Provider Logo Icons & Model Mapping
 * ==========================================================
 *
 * Brand-colored SVG logos for AI model providers.
 * Each logo uses its official brand color for instant recognition.
 *
 * ADDING A NEW PROVIDER:
 * 1. Add an SVG component below with source URL comment
 * 2. Add an entry to PROVIDER_REGISTRY with the logo and display name
 * 3. Add pattern(s) to MODEL_PATTERNS that map model keys to the provider
 *
 * ADDING A NEW MODEL (existing provider):
 * - Just add a pattern to MODEL_PATTERNS. No other changes needed.
 */

import type { ComponentType } from 'react'

interface LogoProps {
  className?: string
}

// =============================================================================
// SVG LOGO COMPONENTS
// =============================================================================

/** Anthropic — angular "A" mark (Claude, Haiku, Opus, Sonnet)
 * SVG source: https://simpleicons.org/icons/anthropic.svg
 * Brand color: #D4A27F (warm tan) */
export function AnthropicLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#D4A27F" aria-label="Anthropic">
      <path d="M17.3041 3.541h-3.6718l6.696 16.918H24Zm-10.6082 0L0 20.459h3.7442l1.3693-3.5527h7.0052l1.3693 3.5528h3.7442L10.5363 3.5409Zm-.3712 10.2232 2.2914-5.9456 2.2914 5.9456Z" />
    </svg>
  )
}

/** OpenAI — hexagonal knot mark (GPT-4, o1, o3, etc.)
 * SVG source: https://icons.getbootstrap.com/icons/openai/
 * Uses currentColor (monochrome — adapts to theme) */
export function OpenAILogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-label="OpenAI">
      <path d="M14.949 6.547a3.94 3.94 0 0 0-.348-3.273 4.11 4.11 0 0 0-4.4-1.934A4.1 4.1 0 0 0 8.423.2 4.15 4.15 0 0 0 6.305.086a4.1 4.1 0 0 0-1.891.948 4.04 4.04 0 0 0-1.158 1.753 4.1 4.1 0 0 0-1.563.679A4 4 0 0 0 .554 4.72a3.99 3.99 0 0 0 .502 4.731 3.94 3.94 0 0 0 .346 3.274 4.11 4.11 0 0 0 4.402 1.933c.382.425.852.764 1.377.995.526.231 1.095.35 1.67.346 1.78.002 3.358-1.132 3.901-2.804a4.1 4.1 0 0 0 1.563-.68 4 4 0 0 0 1.14-1.253 3.99 3.99 0 0 0-.506-4.716m-6.097 8.406a3.05 3.05 0 0 1-1.945-.694l.096-.054 3.23-1.838a.53.53 0 0 0 .265-.455v-4.49l1.366.778q.02.011.025.035v3.722c-.003 1.653-1.361 2.992-3.037 2.996m-6.53-2.75a2.95 2.95 0 0 1-.36-2.01l.095.057L5.29 12.09a.53.53 0 0 0 .527 0l3.949-2.246v1.555a.05.05 0 0 1-.022.041L6.473 13.3c-1.454.826-3.311.335-4.15-1.098m-.85-6.94A3.02 3.02 0 0 1 3.07 3.949v3.785a.51.51 0 0 0 .262.451l3.93 2.237-1.366.779a.05.05 0 0 1-.048 0L2.585 9.342a2.98 2.98 0 0 1-1.113-4.094zm11.216 2.571L8.747 5.576l1.362-.776a.05.05 0 0 1 .048 0l3.265 1.86a3 3 0 0 1 1.173 1.207 2.96 2.96 0 0 1-.27 3.2 3.05 3.05 0 0 1-1.36.997V8.279a.52.52 0 0 0-.276-.445m1.36-2.015-.097-.057-3.226-1.855a.53.53 0 0 0-.53 0L6.249 6.153V4.598a.04.04 0 0 1 .019-.04L9.533 2.7a3.07 3.07 0 0 1 3.257.139c.474.325.843.778 1.066 1.303.223.526.289 1.103.191 1.664zM5.503 8.575 4.139 7.8a.05.05 0 0 1-.026-.037V4.049c0-.57.166-1.127.476-1.607s.752-.864 1.275-1.105a3.08 3.08 0 0 1 3.234.41l-.096.054-3.23 1.838a.53.53 0 0 0-.265.455zm.742-1.577 1.758-1 1.762 1v2l-1.755 1-1.762-1z" />
    </svg>
  )
}

/** Meta — infinity loop mark
 * SVG source: https://simpleicons.org/icons/meta.svg
 * Brand color: #0081FB (Meta blue)
 * Kept in registry for future Meta-branded models */
export function MetaLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#0081FB" aria-label="Meta">
      <path d="M6.915 4.03c-1.968 0-3.683 1.28-4.871 3.113C.704 9.208 0 11.883 0 14.449c0 .706.07 1.369.21 1.973a6.624 6.624 0 0 0 .265.86 5.297 5.297 0 0 0 .371.761c.696 1.159 1.818 1.927 3.593 1.927 1.497 0 2.633-.671 3.965-2.444.76-1.012 1.144-1.626 2.663-4.32l.756-1.339.186-.325c.061.1.121.196.183.3l2.152 3.595c.724 1.21 1.665 2.556 2.47 3.314 1.046.987 1.992 1.22 3.06 1.22 1.075 0 1.876-.355 2.455-.843a3.743 3.743 0 0 0 .81-.973c.542-.939.861-2.127.861-3.745 0-2.72-.681-5.357-2.084-7.45-1.282-1.912-2.957-2.93-4.716-2.93-1.047 0-2.088.467-3.053 1.308-.652.57-1.257 1.29-1.82 2.05-.69-.875-1.335-1.547-1.958-2.056-1.182-.966-2.315-1.303-3.454-1.303zm10.16 2.053c1.147 0 2.188.758 2.992 1.999 1.132 1.748 1.647 4.195 1.647 6.4 0 1.548-.368 2.9-1.839 2.9-.58 0-1.027-.23-1.664-1.004-.496-.601-1.343-1.878-2.832-4.358l-.617-1.028a44.908 44.908 0 0 0-1.255-1.98c.07-.109.141-.224.211-.327 1.12-1.667 2.118-2.602 3.358-2.602zm-10.201.553c1.265 0 2.058.791 2.675 1.446.307.327.737.871 1.234 1.579l-1.02 1.566c-.757 1.163-1.882 3.017-2.837 4.338-1.191 1.649-1.81 1.817-2.486 1.817-.524 0-1.038-.237-1.383-.794-.263-.426-.464-1.13-.464-2.046 0-2.221.63-4.535 1.66-6.088.454-.687.964-1.226 1.533-1.533a2.264 2.264 0 0 1 1.088-.285z" />
    </svg>
  )
}

/** Ollama — llama face mark (local Llama models)
 * SVG source: https://simpleicons.org/icons/ollama.svg
 * Brand color: #F7F7F8 (Ollama off-white, matching their dark-background branding) */
export function OllamaLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#F7F7F8" aria-label="Ollama">
      <path d="M16.361 10.26a.894.894 0 0 0-.558.47l-.072.148.001.207c0 .193.004.217.059.353.076.193.152.312.291.448.24.238.51.3.872.205a.86.86 0 0 0 .517-.436.752.752 0 0 0 .08-.498c-.064-.453-.33-.782-.724-.897a1.06 1.06 0 0 0-.466 0zm-9.203.005c-.305.096-.533.32-.65.639a1.187 1.187 0 0 0-.06.52c.057.309.31.59.598.667.362.095.632.033.872-.205.14-.136.215-.255.291-.448.055-.136.059-.16.059-.353l.001-.207-.072-.148a.894.894 0 0 0-.565-.472 1.02 1.02 0 0 0-.474.007Zm4.184 2c-.131.071-.223.25-.195.383.031.143.157.288.353.407.105.063.112.072.117.136.004.038-.01.146-.029.243-.02.094-.036.194-.036.222.002.074.07.195.143.253.064.052.076.054.255.059.164.005.198.001.264-.03.169-.082.212-.234.15-.525-.052-.243-.042-.28.087-.355.137-.08.281-.219.324-.314a.365.365 0 0 0-.175-.48.394.394 0 0 0-.181-.033c-.126 0-.207.03-.355.124l-.085.053-.053-.032c-.219-.13-.259-.145-.391-.143a.396.396 0 0 0-.193.032zm.39-2.195c-.373.036-.475.05-.654.086-.291.06-.68.195-.951.328-.94.46-1.589 1.226-1.787 2.114-.04.176-.045.234-.045.53 0 .294.005.357.043.524.264 1.16 1.332 2.017 2.714 2.173.3.033 1.596.033 1.896 0 1.11-.125 2.064-.727 2.493-1.571.114-.226.169-.372.22-.602.039-.167.044-.23.044-.523 0-.297-.005-.355-.045-.531-.288-1.29-1.539-2.304-3.072-2.497a6.873 6.873 0 0 0-.855-.031zm.645.937a3.283 3.283 0 0 1 1.44.514c.223.148.537.458.671.662.166.251.26.508.303.82.02.143.01.251-.043.482-.08.345-.332.705-.672.957a3.115 3.115 0 0 1-.689.348c-.382.122-.632.144-1.525.138-.582-.006-.686-.01-.853-.042-.57-.107-1.022-.334-1.35-.68-.264-.28-.385-.535-.45-.946-.03-.192.025-.509.137-.776.136-.326.488-.73.836-.963.403-.269.934-.46 1.422-.512.187-.02.586-.02.773-.002zm-5.503-11a1.653 1.653 0 0 0-.683.298C5.617.74 5.173 1.666 4.985 2.819c-.07.436-.119 1.04-.119 1.503 0 .544.064 1.24.155 1.721.02.107.031.202.023.208a8.12 8.12 0 0 1-.187.152 5.324 5.324 0 0 0-.949 1.02 5.49 5.49 0 0 0-.94 2.339 6.625 6.625 0 0 0-.023 1.357c.091.78.325 1.438.727 2.04l.13.195-.037.064c-.269.452-.498 1.105-.605 1.732-.084.496-.095.629-.095 1.294 0 .67.009.803.088 1.266.095.555.288 1.143.503 1.534.071.128.243.393.264.407.007.003-.014.067-.046.141a7.405 7.405 0 0 0-.548 1.873c-.062.417-.071.552-.071.991 0 .56.031.832.148 1.279L3.42 24h1.478l-.05-.091c-.297-.552-.325-1.575-.068-2.597.117-.472.25-.819.498-1.296l.148-.29v-.177c0-.165-.003-.184-.057-.293a.915.915 0 0 0-.194-.25 1.74 1.74 0 0 1-.385-.543c-.424-.92-.506-2.286-.208-3.451.124-.486.329-.918.544-1.154a.787.787 0 0 0 .223-.531c0-.195-.07-.355-.224-.522a3.136 3.136 0 0 1-.817-1.729c-.14-.96.114-2.005.69-2.834.563-.814 1.353-1.336 2.237-1.475.199-.033.57-.028.776.01.226.04.367.028.512-.041.179-.085.268-.19.374-.431.093-.215.165-.333.36-.576.234-.29.46-.489.822-.729.413-.27.884-.467 1.352-.561.17-.035.25-.04.569-.04.319 0 .398.005.569.04a4.07 4.07 0 0 1 1.914.997c.117.109.398.457.488.602.034.057.095.177.132.267.105.241.195.346.374.43.14.068.286.082.503.045.343-.058.607-.053.943.016 1.144.23 2.14 1.173 2.581 2.437.385 1.108.276 2.267-.296 3.153-.097.15-.193.27-.333.419-.301.322-.301.722-.001 1.053.493.539.801 1.866.708 3.036-.062.772-.26 1.463-.533 1.854a2.096 2.096 0 0 1-.224.258.916.916 0 0 0-.194.25c-.054.109-.057.128-.057.293v.178l.148.29c.248.476.38.823.498 1.295.253 1.008.231 2.01-.059 2.581a.845.845 0 0 0-.044.098c0 .006.329.009.732.009h.73l.02-.074.036-.134c.019-.076.057-.3.088-.516.029-.217.029-1.016 0-1.258-.11-.875-.295-1.57-.597-2.226-.032-.074-.053-.138-.046-.141.008-.005.057-.074.108-.152.376-.569.607-1.284.724-2.228.031-.26.031-1.378 0-1.628-.083-.645-.182-1.082-.348-1.525a6.083 6.083 0 0 0-.329-.7l-.038-.064.131-.194c.402-.604.636-1.262.727-2.04a6.625 6.625 0 0 0-.024-1.358 5.512 5.512 0 0 0-.939-2.339 5.325 5.325 0 0 0-.95-1.02 8.097 8.097 0 0 1-.186-.152.692.692 0 0 1 .023-.208c.208-1.087.201-2.443-.017-3.503-.19-.924-.535-1.658-.98-2.082-.354-.338-.716-.482-1.15-.455-.996.059-1.8 1.205-2.116 3.01a6.805 6.805 0 0 0-.097.726c0 .036-.007.066-.015.066a.96.96 0 0 1-.149-.078A4.857 4.857 0 0 0 12 3.03c-.832 0-1.687.243-2.456.698a.958.958 0 0 1-.148.078c-.008 0-.015-.03-.015-.066a6.71 6.71 0 0 0-.097-.725C8.997 1.392 8.337.319 7.46.048a2.096 2.096 0 0 0-.585-.041Zm.293 1.402c.248.197.523.759.682 1.388.03.113.06.244.069.292.007.047.026.152.041.233.067.365.098.76.102 1.24l.002.475-.12.175-.118.178h-.278c-.324 0-.646.041-.954.124l-.238.06c-.033.007-.038-.003-.057-.144a8.438 8.438 0 0 1 .016-2.323c.124-.788.413-1.501.696-1.711.067-.05.079-.049.157.013zm9.825-.012c.17.126.358.46.498.888.28.854.36 2.028.212 3.145-.019.14-.024.151-.057.144l-.238-.06a3.693 3.693 0 0 0-.954-.124h-.278l-.119-.178-.119-.175.002-.474c.004-.669.066-1.19.214-1.772.157-.623.434-1.185.68-1.382.078-.062.09-.063.159-.012z" />
    </svg>
  )
}

/** Gemini — sparkle/star mark (Google Gemini models)
 * SVG source: https://raw.githubusercontent.com/simple-icons/simple-icons/master/icons/googlegemini.svg
 * Uses the official Gemini blue gradient. */
export function GeminiLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="Gemini">
      <defs>
        <linearGradient id="gemini-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#1A73E8" />
          <stop offset="100%" stopColor="#6C47FF" />
        </linearGradient>
      </defs>
      <path fill="url(#gemini-gradient)" d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81" />
    </svg>
  )
}

/** Gemma — sparkle/star mark (Google's open-weight Gemma models)
 * Same sparkle shape as Gemini but with purple-to-blue gradient to differentiate. */
export function GemmaLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="Gemma">
      <defs>
        <linearGradient id="gemma-gradient" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#8B5CF6" />
          <stop offset="100%" stopColor="#3B82F6" />
        </linearGradient>
      </defs>
      <path fill="url(#gemma-gradient)" d="M11.04 19.32Q12 21.51 12 24q0-2.49.93-4.68.96-2.19 2.58-3.81t3.81-2.55Q21.51 12 24 12q-2.49 0-4.68-.93a12.3 12.3 0 0 1-3.81-2.58 12.3 12.3 0 0 1-2.58-3.81Q12 2.49 12 0q0 2.49-.96 4.68-.93 2.19-2.55 3.81a12.3 12.3 0 0 1-3.81 2.58Q2.49 12 0 12q2.49 0 4.68.96 2.19.93 3.81 2.55t2.55 3.81" />
    </svg>
  )
}

/** Google — "G" mark (Google Search and other Google services, NOT Gemini)
 * SVG source: https://developers.google.com/identity/branding-guidelines */
export function GoogleLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="Google">
      <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4"/>
      <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853"/>
      <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" fill="#FBBC05"/>
      <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335"/>
    </svg>
  )
}

/** Mistral AI — pixelated "M" mark
 * SVG source: https://github.com/simple-icons/simple-icons (mistralai.svg)
 * Brand colors: #FF8205 (Mistral orange, from official brand guidelines) */
export function MistralLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#FF8205" aria-label="Mistral">
      <path d="M17.143 3.429v3.428h-3.429v3.429h-3.428V6.857H6.857V3.43H3.43v13.714H0v3.428h10.286v-3.428H6.857v-3.429h3.429v3.429h3.429v-3.429h3.428v3.429h-3.428v3.428H24v-3.428h-3.43V3.429z" />
    </svg>
  )
}

/** Cohere — organic coral blob mark (Command models)
 * Brand color: #FF7759 (Cohere coral, from Pentagram brand identity)
 * The coral blob/dot is Cohere's most recognizable mark. */
export function CohereLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#FF7759" aria-label="Cohere">
      <path d="M12 3C7.03 3 3 7.03 3 12s4.03 9 9 9c1.62 0 3.14-.43 4.45-1.18-.81-.93-1.3-2.14-1.3-3.47 0-2.9 2.35-5.25 5.25-5.25.31 0 .62.03.91.08A8.97 8.97 0 0 0 12 3zm0 3.6c2.98 0 5.4 2.42 5.4 5.4s-2.42 5.4-5.4 5.4-5.4-2.42-5.4-5.4S9.02 6.6 12 6.6z" />
    </svg>
  )
}

/** xAI — "X" mark (Grok models)
 * SVG source: https://github.com/simple-icons/simple-icons (x.svg)
 * Brand color: currentColor (adapts to theme, xAI uses black/white) */
export function XAILogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="currentColor" aria-label="xAI">
      <path d="M14.234 10.162 22.977 0h-2.072l-7.591 8.824L7.251 0H.258l9.168 13.343L.258 24H2.33l8.016-9.318L16.749 24h6.993zm-2.837 3.299-.929-1.329L3.076 1.56h3.182l5.965 8.532.929 1.329 7.754 11.09h-3.182z" />
    </svg>
  )
}

/** Perplexity — geometric folded shape
 * SVG source: https://github.com/simple-icons/simple-icons (perplexity.svg)
 * Brand color: #20808D (Perplexity turquoise, from official brand) */
export function PerplexityLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#20808D" aria-label="Perplexity">
      <path d="M22.3977 7.0896h-2.3106V.0676l-7.5094 6.3542V.1577h-1.1554v6.1966L4.4904 0v7.0896H1.6023v10.3976h2.8882V24l6.932-6.3591v6.2005h1.1554v-6.0469l6.9318 6.1807v-6.4879h2.8882V7.0896zm-3.4657-4.531v4.531h-5.355l5.355-4.531zm-13.2862.0676 4.8691 4.4634H5.6458V2.6262zM2.7576 16.332V8.245h7.8476l-6.1149 6.1147v1.9723H2.7576zm2.8882 5.0404v-3.8852h.0001v-2.6488l5.7763-5.7764v7.0111l-5.7764 5.2993zm12.7086.0248-5.7766-5.1509V9.0618l5.7766 5.7766v6.5588zm2.8882-5.0652h-1.733v-1.9723L13.3948 8.245h7.8478v8.087z" />
    </svg>
  )
}

/** Microsoft — four-square mark (Phi, MAI models)
 * Brand color: official 4-color treatment */
export function MicrosoftLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="Microsoft">
      <rect x="1" y="1" width="10" height="10" fill="#F25022"/>
      <rect x="13" y="1" width="10" height="10" fill="#7FBA00"/>
      <rect x="1" y="13" width="10" height="10" fill="#00A4EF"/>
      <rect x="13" y="13" width="10" height="10" fill="#FFB900"/>
    </svg>
  )
}

/** NVIDIA — "eye" mark (Nemotron, etc.)
 * SVG source: https://github.com/simple-icons/simple-icons (nvidia.svg)
 * Brand color: #76B900 (NVIDIA green) */
export function NvidiaLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#76B900" aria-label="NVIDIA">
      <path d="M8.948 8.798v-1.43a6.7 6.7 0 0 1 .424-.018c3.922-.124 6.493 3.374 6.493 3.374s-2.774 3.851-5.75 3.851c-.398 0-.787-.062-1.158-.185v-4.346c1.528.185 1.837.857 2.747 2.385l2.04-1.714s-1.492-1.952-4-1.952a6.016 6.016 0 0 0-.796.035m0-4.735v2.138l.424-.027c5.45-.185 9.01 4.47 9.01 4.47s-4.08 4.964-8.33 4.964c-.37 0-.733-.035-1.095-.097v1.325c.3.035.61.062.91.062 3.957 0 6.82-2.023 9.593-4.408.459.371 2.34 1.263 2.73 1.652-2.633 2.208-8.772 3.984-12.253 3.984-.335 0-.653-.018-.971-.053v1.864H24V4.063zm0 10.326v1.131c-3.657-.654-4.673-4.46-4.673-4.46s1.758-1.944 4.673-2.262v1.237H8.94c-1.528-.186-2.73 1.245-2.73 1.245s.68 2.412 2.739 3.11M2.456 10.9s2.164-3.197 6.5-3.533V6.201C4.153 6.59 0 10.653 0 10.653s2.35 6.802 8.948 7.42v-1.237c-4.84-.6-6.492-5.936-6.492-5.936z" />
    </svg>
  )
}

/** AI21 Labs — "21" mark (Jamba models)
 * Brand color: #EC5B2A (AI21 coral/orange, from their official branding) */
export function AI21Logo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#EC5B2A" aria-label="AI21">
      <path d="M4 6h3v12H4V6zm5 0h3l4 7v-7h3v12h-3l-4-7v7H9V6z" />
    </svg>
  )
}

/** Amazon — arrow/smile mark (Nova models)
 * Brand color: #FF9900 (Amazon orange) */
export function AmazonLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#FF9900" aria-label="Amazon">
      <path d="M13.958 10.09c0 1.232.029 2.256-.591 3.351-.502.891-1.301 1.438-2.186 1.438-1.214 0-1.922-.924-1.922-2.292 0-2.692 2.415-3.182 4.7-3.182v.685zm3.186 7.705a.66.66 0 0 1-.753.077c-1.06-.879-1.247-1.286-1.827-2.124-1.748 1.783-2.985 2.317-5.249 2.317C6.886 18.065 5 16.554 5 13.855c0-2.109 1.143-3.545 2.77-4.248 1.412-.611 3.384-.72 4.893-.889v-.331c0-.611.047-1.333-.312-1.861-.314-.473-.916-.666-1.449-.666-1.003 0-1.888.525-2.103 1.58-.047.24-.218.473-.455.486L5.93 7.635c-.21-.047-.443-.217-.384-.54C6.17 3.965 9.138 3 11.773 3c1.36 0 3.138.361 4.213 1.391C17.368 5.643 17.144 7.535 17.144 9.38v3.967c0 1.198.496 1.725.963 2.372.166.233.202.513-.01.687-.531.445-1.475 1.268-1.993 1.731l-.96-.341z" />
      <path d="M21.73 19.128c-2.16 1.598-5.295 2.448-7.993 2.448-3.783 0-7.189-1.398-9.767-3.726-.202-.183-.022-.432.222-.29 2.783 1.618 6.223 2.594 9.78 2.594 2.398 0 5.033-.497 7.458-1.527.365-.155.672.241.3.501z" />
    </svg>
  )
}

/** Moonshot AI — crescent moon mark (Kimi models)
 * Brand color: #5B5FC7 (Kimi blue-purple) */
export function MoonshotLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#5B5FC7" aria-label="Moonshot">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-1.85 0-3.55-.63-4.9-1.69C8.89 16.5 11.35 15 14 15c1.58 0 3.03.57 4.15 1.52A7.963 7.963 0 0 1 12 20zm5.84-3.55A7.456 7.456 0 0 0 14 15a7.44 7.44 0 0 0-5.32 2.22A7.95 7.95 0 0 1 4 12c0-4.41 3.59-8 8-8 1.85 0 3.55.63 4.9 1.69A9.93 9.93 0 0 0 14 9c-1.58 0-3.03.57-4.15 1.52A7.963 7.963 0 0 0 17.84 16.45z" />
    </svg>
  )
}

/** Baidu — bear paw mark (ERNIE models)
 * SVG source: https://github.com/simple-icons/simple-icons (baidu.svg)
 * Brand color: #2319DC (Baidu blue) */
export function BaiduLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#2319DC" aria-label="Baidu">
      <path d="M9.154 0C7.71 0 6.54 1.658 6.54 3.707c0 2.051 1.171 3.71 2.615 3.71 1.446 0 2.614-1.659 2.614-3.71C11.768 1.658 10.6 0 9.154 0zm7.025.594C14.86.58 13.347 2.589 13.2 3.927c-.187 1.745.25 3.487 2.179 3.735 1.933.25 3.175-1.806 3.422-3.364.252-1.555-.995-3.364-2.362-3.674a1.218 1.218 0 0 0-.261-.03zM3.582 5.535a2.811 2.811 0 0 0-.156.008c-2.118.19-2.428 3.24-2.428 3.24-.287 1.41.686 4.425 3.297 3.864 2.617-.561 2.262-3.68 2.183-4.362-.125-1.018-1.292-2.773-2.896-2.75zm16.534 1.753c-2.308 0-2.617 2.119-2.617 3.616 0 1.43.121 3.425 2.988 3.362 2.867-.063 2.553-3.238 2.553-3.988 0-.745-.62-2.99-2.924-2.99zm-8.264 2.478c-1.424.014-2.708.925-3.323 1.947-1.118 1.868-2.863 3.05-3.112 3.363-.25.309-3.61 2.116-2.864 5.42.746 3.301 3.365 3.237 3.365 3.237s1.93.19 4.171-.31c2.24-.495 4.17.123 4.17.123s5.233 1.748 6.665-1.616c1.43-3.364-.808-5.109-.808-5.109s-2.99-2.306-4.736-4.798c-1.072-1.665-2.348-2.268-3.528-2.257z" />
    </svg>
  )
}

/** Inflection AI — Pi mark (Pi models)
 * Brand color: #EF6530 (Inflection coral) */
export function InflectionLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#EF6530" aria-label="Inflection">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-.5 14.5h-1V9h1v7.5zm3 0h-1V11h1v5.5zm0-6.5h-1v-1h1v1z" />
    </svg>
  )
}

/** IBM — 8-bar striped "IBM" letters (Granite models)
 * Brand color: #0F62FE (IBM blue 60, from IBM Design Language) */
export function IBMLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#0F62FE" aria-label="IBM">
      <path d="M1 4h7v1.5H1zm9 0h4v1.5h-4zm6 0h7v1.5h-7zM1 6.5h7V8H1zm9 0h4V8h-4zm6 0h7V8h-7zM3 9h3v1.5H3zm7 0h4v1.5h-4zm6 0h3v1.5h-3zM3 11.5h3V13H3zm7 0h4V13h-4zm6 0h3V13h-3zM3 14h3v1.5H3zm7 0h4v1.5h-4zm6 0h3v1.5h-3zM3 16.5h3V18H3zm7 0h4V18h-4zm6 0h3V18h-3zM1 19h7v1.5H1zm9 0h4v1.5h-4zm6 0h7v1.5h-7z" />
    </svg>
  )
}

/** Tencent — stylized mark (Hunyuan models)
 * Brand color: #1DA1F2 (Tencent blue) */
export function TencentLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#1DA1F2" aria-label="Tencent">
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 14l-1.41-1.41L10.17 13H6v-2h4.17l-1.59-1.59L10 8l4 4-4 4z" />
    </svg>
  )
}

/** ByteDance — note/music mark (Seed, UI-TARS models)
 * Brand color: #325AB4 (ByteDance blue) */
export function ByteDanceLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#325AB4" aria-label="ByteDance">
      <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.18l7.5 3.75V12L12 15.75 4.5 12V7.93L12 4.18zM4.5 13.82L12 17.57v2.25l-7.5-3.75v-2.25zm15 0v2.25l-7.5 3.75v-2.25l7.5-3.75z" />
    </svg>
  )
}

/** Alibaba — cloud mark (Tongyi models)
 * Brand color: #FF6A00 (Alibaba orange) */
export function AlibabaLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#FF6A00" aria-label="Alibaba">
      <path d="M5.6 8C3.6 8 2 9.6 2 11.6v.8C2 14.4 3.6 16 5.6 16h12.8c2 0 3.6-1.6 3.6-3.6v-.8c0-2-1.6-3.6-3.6-3.6H5.6zm1.2 2.4h10.4c.66 0 1.2.54 1.2 1.2s-.54 1.2-1.2 1.2H6.8c-.66 0-1.2-.54-1.2-1.2s.54-1.2 1.2-1.2z" />
    </svg>
  )
}

/** DuckDuckGo — duck mascot mark (web search provider)
 * SVG source: https://raw.githubusercontent.com/simple-icons/simple-icons/master/icons/duckduckgo.svg
 * The compound path uses even-odd fill: the duck details are cutouts in the red-orange circle.
 * A white circle behind the path makes the cutouts render as white, matching the real logo. */
export function DuckDuckGoLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="DuckDuckGo">
      <circle cx="12" cy="12" r="12" fill="white" />
      <path fill="#DE5833" d="M12 0C5.37 0 0 5.37 0 12s5.37 12 12 12 12-5.37 12-12S18.63 0 12 0zm0 .984C18.083.984 23.016 5.916 23.016 12S18.084 23.016 12 23.016.984 18.084.984 12C.984 5.917 5.916.984 12 .984zm0 .938C6.434 1.922 1.922 6.434 1.922 12c0 4.437 2.867 8.205 6.85 9.55-.237-.82-.776-2.753-1.6-6.052-1.184-4.741-2.064-8.606 2.379-9.813.047-.011.064-.064.03-.093-.514-.467-1.382-.548-2.233-.38a.06.06 0 0 1-.07-.058c0-.011 0-.023.011-.035.205-.286.572-.507.822-.64a1.843 1.843 0 0 0-.607-.335c-.059-.022-.059-.12-.006-.144.006-.006.012-.012.024-.012 1.749-.233 3.586.292 4.49 1.448.011.011.023.017.035.023 2.968.635 3.509 4.837 3.328 5.998a9.607 9.607 0 0 0 2.346-.576c.746-.286 1.008-.222 1.101-.053.1.193-.018.513-.28.81-.496.567-1.393 1.01-2.974 1.137-.546.044-1.029.024-1.445.006-.789-.035-1.339-.059-1.633.39-.192.298-.041.998 1.487 1.22 1.09.157 2.078.047 2.798-.034.643-.07 1.073-.118 1.172.069.21.402-.996 1.207-3.066 1.224-.158 0-.315-.006-.467-.011-1.283-.065-2.227-.414-2.816-.735a.094.094 0 0 1-.035-.017c-.105-.059-.31.045-.188.267.07.134.444.478 1.004.776-.058.466.087 1.184.338 2l.088-.016c.041-.009.087-.019.134-.025.507-.082.775.012.926.175.717-.536 1.913-1.294 2.03-1.154.583.694.66 2.332.53 2.99-.004.012-.017.024-.04.035-.274.117-1.783-.296-1.783-.511-.059-1.075-.26-1.173-.493-1.225h-.156c.006.006.012.018.018.03l.052.12c.093.257.24 1.063.13 1.26-.112.199-.835.297-1.284.303-.443.006-.543-.158-.637-.408-.07-.204-.103-.675-.103-.95a.857.857 0 0 1 .012-.216c-.134.058-.333.193-.397.281-.017.262-.017.682.123 1.149.07.221-1.518 1.164-1.74.99-.227-.181-.634-1.952-.459-2.67-.187.017-.338.075-.42.191-.367.508.093 2.933.582 3.248.257.169 1.54-.553 2.176-1.095.105.145.305.158.553.158.326-.012.782-.06 1.103-.158.192.45.423.972.613 1.388 4.47-1.032 7.803-5.037 7.803-9.82 0-5.566-4.512-10.078-10.078-10.078zm1.791 5.646c-.42 0-.678.146-.795.332-.023.047.047.094.094.07.14-.075.357-.161.701-.156.328.006.516.09.67.159l.023.01c.041.017.088-.03.059-.065-.134-.18-.332-.35-.752-.35zm-5.078.198a1.24 1.24 0 0 0-.522.082c-.454.169-.67.526-.67.76 0 .051.112.057.141.011.081-.123.21-.31.617-.478.408-.17.73-.146.951-.094.047.012.083-.041.041-.07a.989.989 0 0 0-.558-.211zm5.434 1.423a.651.651 0 0 0-.655.647.652.652 0 0 0 1.307 0 .646.646 0 0 0-.652-.647zm.283.262h.008a.17.17 0 0 1 .17.17c0 .093-.077.17-.17.17a.17.17 0 0 1-.17-.17c0-.09.072-.165.162-.17zm-5.358.076a.752.752 0 0 0-.758.758c0 .42.338.758.758.758s.758-.337.758-.758a.756.756 0 0 0-.758-.758zm.328.303h.01c.112 0 .2.089.2.2 0 .11-.088.197-.2.197a.195.195 0 0 1-.197-.198c0-.107.082-.194.187-.199z" />
    </svg>
  )
}

/** Tavily — branching arrows mark (AI-optimized web search provider)
 * Traced from Tavily's actual 32x32 favicon: one upward arrow with two
 * branches (right and down-left) forking from the shaft midpoint. */
export function TavilyLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-label="Tavily">
      <rect x="1" y="1" width="22" height="22" rx="4.5" fill="#333333" />
      <g fill="white">
        {/* Main upward arrow: arrowhead + shaft */}
        <path d="M10,4.5 L7,8.5 L8.8,8.5 L8.8,13 L11.2,13 L11.2,8.5 L13,8.5 Z" />
        {/* Right branch arrow */}
        <g transform="translate(12,13.5) rotate(90)">
          <path d="M0,-4.5 L-2,0 L-0.8,0 L-0.8,3 L0.8,3 L0.8,0 L2,0 Z" />
        </g>
        {/* Down-left branch arrow */}
        <g transform="translate(9.5,14.5) rotate(145)">
          <path d="M0,-4.5 L-2,0 L-0.8,0 L-0.8,3 L0.8,3 L0.8,0 L2,0 Z" />
        </g>
      </g>
    </svg>
  )
}

/** DeepSeek — whale/fish mark (DeepSeek R1, Coder, etc.)
 * SVG source: https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/deepseek.svg
 * Brand color: #4D6BFE (DeepSeek blue) */
export function DeepSeekLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 377.1 277.86" fill="#4D6BFE" aria-label="DeepSeek">
      <path d="M373.15,23.32c-4-1.95-5.72,1.77-8.06,3.66-.79.62-1.47,1.43-2.14,2.14-5.85,6.26-12.67,10.36-21.57,9.86-13.04-.71-24.16,3.38-33.99,13.37-2.09-12.31-9.04-19.66-19.6-24.38-5.54-2.45-11.13-4.9-14.99-10.23-2.71-3.78-3.44-8-4.81-12.16-.85-2.51-1.72-5.09-4.6-5.52-3.13-.5-4.36,2.14-5.58,4.34-4.93,8.99-6.82,18.92-6.65,28.97.43,22.58,9.97,40.56,28.89,53.37,2.16,1.46,2.71,2.95,2.03,5.09-1.29,4.4-2.82,8.68-4.19,13.09-.85,2.82-2.14,3.44-5.15,2.2-10.39-4.34-19.37-10.76-27.29-18.55-13.46-13.02-25.63-27.41-40.81-38.67-3.57-2.64-7.12-5.09-10.81-7.41-15.49-15.07,2.03-27.45,6.08-28.9,4.25-1.52,1.47-6.79-12.23-6.73-13.69.06-26.24,4.65-42.21,10.76-2.34.93-4.79,1.61-7.32,2.14-14.5-2.73-29.55-3.35-45.29-1.58-29.62,3.32-53.28,17.34-70.68,41.28C1.29,88.2-3.63,120.88,2.39,155c6.33,35.91,24.64,65.68,52.8,88.94,29.18,24.1,62.8,35.91,101.15,33.65,23.29-1.33,49.23-4.46,78.48-29.24,7.38,3.66,15.12,5.12,27.97,6.23,9.89.93,19.41-.5,26.79-2.02,11.55-2.45,10.75-13.15,6.58-15.13-33.87-15.78-26.44-9.36-33.2-14.54,17.21-20.41,43.15-41.59,53.3-110.19.79-5.46.11-8.87,0-13.3-.06-2.67.54-3.72,3.61-4.03,8.48-.96,16.72-3.29,24.28-7.47,21.94-12,30.78-31.69,32.87-55.33.31-3.6-.06-7.35-3.86-9.24ZM181.96,235.97c-32.83-25.83-48.74-34.33-55.31-33.96-6.14.34-5.04,7.38-3.69,11.97,1.41,4.53,3.26,7.66,5.85,11.63,1.78,2.64,3.01,6.57-1.78,9.49-10.57,6.58-28.95-2.2-29.82-2.64-21.38-12.59-39.26-29.24-51.87-52.01-12.16-21.92-19.23-45.43-20.39-70.52-.31-6.08,1.47-8.22,7.49-9.3,7.92-1.46,16.11-1.77,24.03-.62,33.49,4.9,62.01,19.91,85.9,43.63,13.65,13.55,23.97,29.71,34.61,45.49,11.3,16.78,23.48,32.75,38.97,45.84,5.46,4.59,9.83,8.09,14,10.67-12.59,1.4-33.62,1.71-47.99-9.68ZM197.69,134.65c0-2.7,2.15-4.84,4.87-4.84.6,0,1.16.12,1.66.31.67.25,1.29.62,1.77,1.18.87.84,1.36,2.08,1.36,3.35,0,2.7-2.15,4.84-4.85,4.84s-4.81-2.14-4.81-4.84ZM246.55,159.77c-3.13,1.27-6.26,2.39-9.27,2.51-4.67.22-9.77-1.68-12.55-4-4.3-3.6-7.36-5.61-8.67-11.94-.54-2.7-.23-6.85.25-9.24,1.12-5.15-.12-8.44-3.74-11.44-2.96-2.45-6.7-3.1-10.82-3.1-1.54,0-2.95-.68-4-1.24-1.72-.87-3.13-3.01-1.78-5.64.43-.84,2.53-2.92,3.02-3.29,5.58-3.19,12.03-2.14,18,.25,5.54,2.26,9.71,6.42,15.72,12.28,6.16,7.1,7.26,9.09,10.76,14.39,2.76,4.19,5.29,8.47,7.01,13.37,1.04,3.04-.31,5.55-3.94,7.1Z" />
    </svg>
  )
}

/** Z.ai (Zhipu AI) — stylized "Z" mark (GLM models)
 * Z.ai (formerly Zhipu AI) develops the GLM family of models.
 * Brand color: #4F46E5 (indigo, matching their official branding) */
export function ZhipuLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#4F46E5" aria-label="Z.ai">
      <path d="M4 4h16v3.5H10.5L20 16.5V20H4v-3.5h9.5L4 7.5V4z" />
    </svg>
  )
}

/** OpenRouter — routing arrows mark (API aggregator)
 * SVG source: https://cdn.simpleicons.org/openrouter
 * Brand color: #94A3B8 (slate gray) */
export function OpenRouterLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="#94A3B8" aria-label="OpenRouter">
      <path d="M16.778 1.844v1.919q-.569-.026-1.138-.032-.708-.008-1.415.037c-1.93.126-4.023.728-6.149 2.237-2.911 2.066-2.731 1.95-4.14 2.75-.396.223-1.342.574-2.185.798-.841.225-1.753.333-1.751.333v4.229s.768.108 1.61.333c.842.224 1.789.575 2.185.799 1.41.798 1.228.683 4.14 2.75 2.126 1.509 4.22 2.11 6.148 2.236.88.058 1.716.041 2.555.005v1.918l7.222-4.168-7.222-4.17v2.176c-.86.038-1.611.065-2.278.021-1.364-.09-2.417-.357-3.979-1.465-2.244-1.593-2.866-2.027-3.68-2.508.889-.518 1.449-.906 3.822-2.59 1.56-1.109 2.614-1.377 3.978-1.466.667-.044 1.418-.017 2.278.02v2.176L24 6.014Z" />
    </svg>
  )
}

/** Qwen — geometric star mark (Qwen / Alibaba models)
 * SVG source: https://cdn.jsdelivr.net/gh/homarr-labs/dashboard-icons/svg/qwen.svg
 * Brand color: #6C5CE7 (Qwen purple) */
export function QwenLogo({ className }: LogoProps) {
  return (
    <svg className={className} viewBox="27.55 17.52 147.28 145.51" aria-label="Qwen">
      <path d="M174.82 108.75L155.38 75L165.64 57.75C166.46 56.31 166.46 54.53 165.64 53.09L155.38 35.84C154.86 34.91 153.87 34.33 152.78 34.33H114.88L106.14 19.03C105.62 18.1 104.63 17.52 103.54 17.52H83.3C82.21 17.52 81.22 18.1 80.7 19.03L61.26 52.77H41.02C39.93 52.77 38.94 53.35 38.42 54.28L28.16 71.53C27.34 72.97 27.34 74.75 28.16 76.19L45.52 107.5L36.78 122.8C35.96 124.24 35.96 126.02 36.78 127.46L47.04 144.71C47.56 145.64 48.55 146.22 49.64 146.22H87.54L96.28 161.52C96.8 162.45 97.79 163.03 98.88 163.03H119.12C120.21 163.03 121.2 162.45 121.72 161.52L141.16 127.78H158.52C159.61 127.78 160.6 127.2 161.12 126.27L171.38 109.02C172.2 107.58 172.2 105.8 171.38 104.36L174.82 108.75Z" fill="#6C5CE7" />
      <path d="M119.12 163.03H98.88L87.54 144.71H49.64L61.26 126.39H80.7L38.42 55.29H61.26L83.3 19.03L93.56 37.35L83.3 55.29H161.58L151.32 72.54L170.76 106.28H151.32L141.16 88.34L101.18 163.03H119.12Z" fill="white" />
      <path d="M127.86 79.83H76.14L101.18 122.11L127.86 79.83Z" fill="#6C5CE7" />
    </svg>
  )
}

// =============================================================================
// PROVIDER REGISTRY
// =============================================================================
// Central registry of all known providers. To add a new provider:
// 1. Create an SVG component above
// 2. Add it here with a display name

type ProviderKey = 'anthropic' | 'openai' | 'meta' | 'ollama' | 'gemini' | 'gemma' | 'google' | 'qwen' | 'deepseek' | 'zhipu' | 'openrouter' | 'mistral' | 'cohere' | 'xai' | 'perplexity' | 'microsoft' | 'nvidia' | 'ai21' | 'amazon' | 'moonshot' | 'baidu' | 'inflection' | 'ibm' | 'tencent' | 'bytedance' | 'alibaba' | 'duckduckgo' | 'tavily' | 'googlesearch'

interface ProviderInfo {
  name: string
  Logo: ComponentType<LogoProps>
}

const PROVIDER_REGISTRY: Record<ProviderKey, ProviderInfo> = {
  anthropic:    { name: 'Anthropic',    Logo: AnthropicLogo },
  openai:       { name: 'OpenAI',       Logo: OpenAILogo },
  meta:         { name: 'Meta',         Logo: MetaLogo },
  ollama:       { name: 'Ollama',       Logo: OllamaLogo },
  gemini:       { name: 'Gemini',       Logo: GeminiLogo },
  gemma:        { name: 'Gemma',        Logo: GemmaLogo },
  google:       { name: 'Google',       Logo: GoogleLogo },
  qwen:         { name: 'Qwen',         Logo: QwenLogo },
  deepseek:     { name: 'DeepSeek',     Logo: DeepSeekLogo },
  zhipu:        { name: 'Z.ai',         Logo: ZhipuLogo },
  openrouter:   { name: 'OpenRouter',   Logo: OpenRouterLogo },
  mistral:      { name: 'Mistral',      Logo: MistralLogo },
  cohere:       { name: 'Cohere',       Logo: CohereLogo },
  xai:          { name: 'xAI',          Logo: XAILogo },
  perplexity:   { name: 'Perplexity',   Logo: PerplexityLogo },
  microsoft:    { name: 'Microsoft',    Logo: MicrosoftLogo },
  nvidia:       { name: 'NVIDIA',       Logo: NvidiaLogo },
  ai21:         { name: 'AI21',         Logo: AI21Logo },
  amazon:       { name: 'Amazon',       Logo: AmazonLogo },
  moonshot:     { name: 'Moonshot',     Logo: MoonshotLogo },
  baidu:        { name: 'Baidu',        Logo: BaiduLogo },
  inflection:   { name: 'Inflection',   Logo: InflectionLogo },
  ibm:          { name: 'IBM',          Logo: IBMLogo },
  tencent:      { name: 'Tencent',      Logo: TencentLogo },
  bytedance:    { name: 'ByteDance',    Logo: ByteDanceLogo },
  alibaba:      { name: 'Alibaba',      Logo: AlibabaLogo },
  duckduckgo:   { name: 'DuckDuckGo',   Logo: DuckDuckGoLogo },
  tavily:       { name: 'Tavily',       Logo: TavilyLogo },
  googlesearch: { name: 'Google Search', Logo: GoogleLogo },
}

// =============================================================================
// MODEL → PROVIDER MAPPING
// =============================================================================
// Maps model keys/names to providers. Checked in order — first match wins.
// To support a new model, add a pattern here. No other file changes needed.
//
// Patterns are checked with string.includes(), so partial matches work.
// Put more specific patterns before general ones.

const MODEL_PATTERNS: [string, ProviderKey][] = [
  // Anthropic models
  ['claude',    'anthropic'],
  ['haiku',     'anthropic'],
  ['opus',      'anthropic'],
  ['sonnet',    'anthropic'],

  // OpenAI models
  ['gpt',       'openai'],
  ['o1',        'openai'],
  ['o3',        'openai'],
  ['o4',        'openai'],
  ['davinci',   'openai'],
  ['chatgpt',   'openai'],

  // Llama models — use the llama face icon (the MODEL logo, not Meta corporate).
  // Matches both cloud (meta-llama/*, maverick) and local (ollama) Llama models.
  ['maverick',  'ollama'],
  ['meta-llama', 'ollama'],
  ['llama',     'ollama'],
  ['codellama', 'ollama'],

  // Gemini models (uses Gemini sparkle logo, not Google "G")
  ['gemini',    'gemini'],

  // Gemma models
  ['gemma',        'gemma'],
  ['local-vision', 'gemma'],

  // Mistral AI models
  ['mistral',   'mistral'],
  ['mixtral',   'mistral'],
  ['mistralai', 'mistral'],
  ['pixtral',   'mistral'],
  ['codestral', 'mistral'],
  ['ministral', 'mistral'],

  // Cohere models
  ['cohere',    'cohere'],
  ['command-r', 'cohere'],
  ['command-a', 'cohere'],

  // xAI / Grok models
  ['grok',      'xai'],
  ['x-ai',      'xai'],

  // Perplexity models
  ['perplexity', 'perplexity'],
  ['sonar',      'perplexity'],

  // Microsoft models (Phi, MAI, WizardLM)
  ['microsoft', 'microsoft'],
  ['phi-',      'microsoft'],
  ['wizardlm',  'microsoft'],

  // NVIDIA models (Nemotron, etc.)
  ['nvidia',    'nvidia'],
  ['nemotron',  'nvidia'],

  // AI21 models (Jamba)
  ['ai21',      'ai21'],
  ['jamba',     'ai21'],

  // Amazon models (Nova)
  ['amazon',    'amazon'],

  // Moonshot / Kimi models
  ['moonshotai', 'moonshot'],
  ['kimi',       'moonshot'],

  // Baidu models (ERNIE)
  ['baidu',     'baidu'],
  ['ernie',     'baidu'],

  // Inflection models (Pi)
  ['inflection', 'inflection'],

  // IBM models (Granite)
  ['ibm-granite', 'ibm'],
  ['granite',     'ibm'],

  // Tencent models (Hunyuan)
  ['tencent',   'tencent'],
  ['hunyuan',   'tencent'],

  // ByteDance models (Seed, UI-TARS)
  ['bytedance-seed', 'bytedance'],
  ['bytedance',      'bytedance'],
  ['seed-',          'bytedance'],

  // Alibaba models (Tongyi)
  ['alibaba',   'alibaba'],
  ['tongyi',    'alibaba'],

  // Web search providers
  ['duckduckgo', 'duckduckgo'],
  ['tavily',     'tavily'],

  // DeepSeek models
  ['deepseek',  'deepseek'],

  // Z.ai / Zhipu models (GLM family)
  ['glm',       'zhipu'],
  ['z-ai',      'zhipu'],

  // Qwen / Alibaba models (Lakeshore tier)
  ['qwq',       'qwen'],
  ['qwen',      'qwen'],
  ['coder',     'qwen'],
]

// =============================================================================
// LOOKUP FUNCTIONS
// =============================================================================

/**
 * Get provider info for a model key or name.
 *
 * Works with any format:
 *   - Config keys: "cloud-claude", "local-llama", "local-vision", "lakeshore-qwen"
 *   - Raw model names: "llama3.2:3b", "claude-sonnet-4", "gpt-4o"
 *   - Display names: "Claude Sonnet 4", "GPT-4o", "Llama 3.2 3B"
 *
 * Returns null if no provider matches (unknown model).
 */
export function getModelProvider(modelKey: string): ProviderInfo | null {
  const lower = modelKey.toLowerCase()
  for (const [pattern, providerKey] of MODEL_PATTERNS) {
    if (lower.includes(pattern)) {
      return PROVIDER_REGISTRY[providerKey]
    }
  }
  return null
}

/**
 * Convenience component — renders the right provider logo for a model key.
 *
 * Usage:
 *   <ModelLogo model="cloud-claude" className="w-4 h-4" />
 *   <ModelLogo model="llama3.2:3b" className="w-4 h-4" />
 *
 * Renders nothing if the model isn't recognized.
 */
// Deterministic color palette for unknown providers (letter avatars).
// These are hand-picked to be legible on both light and dark backgrounds.
const AVATAR_COLORS = [
  '#E74C3C', '#E67E22', '#F1C40F', '#2ECC71', '#1ABC9C',
  '#3498DB', '#9B59B6', '#E91E63', '#00BCD4', '#FF5722',
  '#795548', '#607D8B', '#8BC34A', '#FF9800', '#673AB7',
]

function getAvatarColor(name: string): string {
  let hash = 0
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash)
  }
  return AVATAR_COLORS[Math.abs(hash) % AVATAR_COLORS.length]
}

function getProviderInitial(model: string): string {
  // Extract provider prefix from catalog IDs like "arcee-ai/trinity-mini"
  if (model.includes('/')) {
    const provider = model.split('/')[0]
    return provider.charAt(0).toUpperCase()
  }
  return model.charAt(0).toUpperCase()
}

export function ModelLogo({ model, className }: { model: string; className?: string }) {
  const provider = getModelProvider(model)
  if (!provider) {
    const initial = getProviderInitial(model)
    const color = getAvatarColor(model.split('/')[0] || model)
    return (
      <svg className={className} viewBox="0 0 24 24" aria-label={model}>
        <circle cx="12" cy="12" r="11" fill={color} />
        <text x="12" y="17" textAnchor="middle" fontSize="13" fontWeight="bold" fill="white" fontFamily="system-ui, sans-serif">{initial}</text>
      </svg>
    )
  }
  const { Logo } = provider
  return <Logo className={className} />
}
