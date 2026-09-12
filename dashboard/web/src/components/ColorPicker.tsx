import { useState } from 'react'
import { PALETTE } from '../lib/api'
import type { CSSProperties } from 'react'

interface Props {
  color: string | null
  onPick: (color: string) => void
  label: string
}

/**
 * A swatch that opens the palette.
 *
 * Only the eight documented colours are offered. They are the ones checked
 * against this surface -- each clears 3:1, and neighbouring pairs stay apart
 * for colour-blind readers -- which a free colour well could not promise.
 */
export function ColorPicker({ color, onPick, label }: Props) {
  const [open, setOpen] = useState(false)

  return (
    <span className="color-picker">
      <button
        type="button"
        className="swatch"
        style={{ '--swatch': color ?? 'var(--muted)' } as CSSProperties}
        aria-label={`Change colour for ${label}`}
        title={`Change colour for ${label}`}
        onClick={(event) => {
          event.stopPropagation()
          setOpen((current) => !current)
        }}
      />
      {open && (
        <span className="swatch-menu">
          {PALETTE.map((option) => (
            <button
              key={option}
              type="button"
              className={option === color ? 'swatch chosen' : 'swatch'}
              style={{ '--swatch': option } as CSSProperties}
              aria-label={option}
              onClick={(event) => {
                event.stopPropagation()
                onPick(option)
                setOpen(false)
              }}
            />
          ))}
        </span>
      )}
    </span>
  )
}
