# Compute — Design Language

Reference document for building pages/apps consistent with the Compute website.

---

## Color Palette

Warm, professional neutrals (stone/taupe family). Not cold or sterile.

| Token | Hex | Usage |
|-------|-----|-------|
| `cpu-white` | `#ffffff` | Default background, surfaces |
| `cpu-cream` | `#fafaf9` | Light section backgrounds, card fills |
| `cpu-light` | `#f5f5f4` | Alternate light surfaces |
| `cpu-silver` | `#e7e5e4` | Subtle backgrounds |
| `cpu-border` | `#d6d3d1` | All borders (applied globally via `* { @apply border-cpu-border }`) |
| `cpu-muted` | `#a8a29e` | Secondary text, labels, icons |
| `cpu-text` | `#57534e` | Primary body text |
| `cpu-dark` | `#1c1917` | Headings, primary button bg |
| `cpu-black` | `#0c0a09` | Dark section backgrounds, button hover |

**Semantic accents** — used sparingly, only for meaning:
- `green-500` / `emerald-500`: success, positive states
- `red-500`: deprecated/problem states
- No brand accent color — the palette is intentionally monochromatic

**Why:** A monochrome stone palette keeps the feel professional and tech-forward without being sterile. Color is reserved for semantic meaning so it always communicates something.

---

## Typography

Three font families, each with a distinct role:

| Role | Font | Where |
|------|------|-------|
| **Display** | DM Sans (medium weight) | All headings h1–h6, hero copy |
| **Body** | Inter | Default body text, descriptions |
| **Mono** | IBM Plex Mono | Labels, code, technical content |

### Type Scale (custom display sizes)

| Class | Size | Line Height | Letter Spacing | Usage |
|-------|------|-------------|----------------|-------|
| `display-2xl` | 5rem (80px) | 1 | -0.025em | Hero headline |
| `display-xl` | 4rem (64px) | 1.05 | -0.025em | Major section headings |
| `display-lg` | 3rem (48px) | 1.1 | -0.02em | Section headings |
| `display-md` | 2.25rem (36px) | 1.2 | -0.015em | Sub-headings |
| `display-sm` | 1.5rem (24px) | 1.3 | -0.01em | Card titles |

All display sizes use `fontWeight: 500`. Tight negative letter-spacing is key to the look — don't omit it.

Body text uses standard Tailwind sizes: `text-lg` for paragraphs, `text-sm` for secondary text, `text-xs` for micro labels.

**Font features:** `font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11'` applied globally to Inter for cleaner letterforms.

**Text selection:** `bg-cpu-dark/10 text-cpu-dark` — subtle, on-brand.

---

## Spacing & Layout

### Containers

Two container classes — use them consistently:

- **`.container-wide`**: `max-w-[1400px] mx-auto px-6 md:px-10 lg:px-16` — most sections
- **`.container-narrow`**: `max-w-4xl mx-auto px-6 md:px-10` — focused/text-heavy content

### Section Rhythm

**`.section-padding`**: `py-20 md:py-28 lg:py-36` — generous vertical breathing room between sections. This spacing is critical to the feel.

### Grid Patterns

Layouts use CSS Grid with a 1px gap trick (`gap-px`) that exposes the border color as dividers:

- **Feature grid**: `grid md:grid-cols-2 lg:grid-cols-4 gap-px` (large features span 2 cols)
- **Hero**: 12-column grid at `lg` — content gets 7 cols, sidebar gets 5
- **How It Works**: `grid md:grid-cols-3 gap-px`
- **Stats**: `grid grid-cols-2 md:grid-cols-4 gap-8 md:divide-x`
- **Footer**: `grid grid-cols-2 md:grid-cols-6 gap-10 lg:gap-16`

**Why the gap-px pattern:** Instead of visible borders on each card, the 1px gap between grid items exposes the page background (or a border color), creating clean divider lines without extra markup.

---

## Component Patterns

### Buttons

Three tiers:

| Class | Style | Hover |
|-------|-------|-------|
| `.btn-primary` | `bg-cpu-dark text-white px-6 py-3 text-sm font-medium` | `bg-cpu-black` |
| `.btn-secondary` | `border border-cpu-dark text-cpu-dark px-6 py-3 text-sm font-medium` | `bg-cpu-dark text-white` (inverts) |
| `.btn-ghost` | `text-cpu-text font-medium text-sm` (no bg/border) | `text-cpu-dark` |

Buttons use `gap-2` for icon support. Arrow icons animate `translate-x-1` on group hover.

### Cards

**`.card`**: `bg-cpu-cream border border-cpu-border p-8`
- Hover: `border-cpu-muted` (subtle border darken)
- Transition: `all duration-300`
- Feature cards shift to `bg-cpu-white` on hover

Cards are intentionally flat — no shadows, no rounded corners (except the terminal card in the hero). The border system carries the visual structure.

### Labels

**`.label`**: `text-xs font-mono uppercase tracking-widest text-cpu-muted`

Used as section eyebrows above headings. The mono + uppercase + wide tracking pattern is a signature element — use it consistently for categorical/metadata text.

### Dividers

**`.divider`**: `h-px bg-cpu-border` — simple horizontal rules.

---

## Visual Effects

### Borders as Structure

The entire design is built on a border system rather than shadows or fills. Every element gets `border-cpu-border` globally. This creates a technical, grid-like feel.

### Header Scroll State

On scroll: `bg-cpu-white/90 backdrop-blur-md border-b` — frosted glass effect.

### Dark Section (Infrastructure)

One section breaks the light palette with `bg-cpu-dark text-cpu-white`. Uses:
- `bg-cpu-white/10` for card backgrounds (glass-on-dark)
- `border-white/10` for dividers
- Green/red accent colors pop against the dark background

**Why:** Creates a visual "break" in the page flow to signal a shift in content tone (technical depth).

### Mask Gradient

**`.mask-gradient-x`**: Horizontal fade on edges — used for the partner logo marquee so items fade in/out smoothly.

---

## Animation & Motion

### Philosophy

Animations are subtle and performance-focused — only `transform` and `opacity` changes. Nothing flashy or bouncy.

### Entry Animations (scroll-triggered via IntersectionObserver)

Sections animate in with a staggered sequence:
1. Label: `fadeIn 0.6s`
2. Heading: `slideUp 0.8s, delay 0.1s`
3. Body text: `slideUp 0.6s, delay 0.2s`
4. CTAs: `slideUp 0.6s, delay 0.3s`
5. Cards: `slideUp 0.5s`, staggered `+0.05–0.15s` per card

### Keyframes

| Name | Effect | Duration |
|------|--------|----------|
| `fadeIn` | opacity 0→1 | 0.8s ease-out |
| `slideUp` | opacity 0→1, translateY 30px→0 | 0.8s ease-out |
| `slideInLeft` | opacity 0→1, translateX -30px→0 | 0.8s ease-out |
| `marquee` | translateX 0→-50% | infinite linear |

### Hover & Interactive

- Transitions: `duration-200` for most interactions, `duration-300` for cards
- Arrow icons: `translate-x-1` on hover
- Dropdown menus (Framer Motion): `opacity 0→1, y -8→0, duration 0.15s`
- Count-up numbers: animate from 0 to value over 2000ms, ease-out cubic

### Marquee (partner logos)

Infinite horizontal scroll: 40s on mobile, 60s on tablet, 80s on large screens. Edges masked with gradient fade.

---

## Responsive Strategy

Mobile-first. Content stacks vertically on mobile and expands to multi-column on larger screens.

| Breakpoint | Key Changes |
|------------|-------------|
| Base (mobile) | Single column, `px-6`, `py-20`, hamburger nav |
| `md` (768px) | 2-column grids, `px-10`, `py-28` |
| `lg` (1024px) | Full multi-column layouts, desktop nav visible, `px-16`, `py-36` |
| `2xl` (1536px) | Marquee slows down (80s). Layout stays the same |

Hero headline scales from `display-xl` on mobile to `display-2xl` at `lg`.

---

## Overall Aesthetic

**Minimal. Warm. Technical.**

- Monochromatic stone palette — no bright brand colors
- Typography-driven hierarchy with tight letter-spacing on display type
- Borders and grid lines as primary structural elements (not shadows or fills)
- Flat cards, no border-radius (square/sharp edges throughout)
- Generous whitespace — sections breathe
- One dark section break for contrast
- Subtle scroll-triggered animations, never distracting
- Mono font for labels/metadata gives a technical, systematic feel

The design avoids common "startup landing page" tropes — no gradient blobs, no glassmorphism, no neon accents. It's closer to editorial/architectural design: quiet confidence through restraint and precision.
