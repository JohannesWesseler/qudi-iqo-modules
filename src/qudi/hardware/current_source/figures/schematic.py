"""Schematic of the current source as built (Nov 2026). Regenerate with
    python schematic.py            (needs schemdraw; writes schematic.svg and schematic.png)
"""
import os

import schemdraw
import schemdraw.elements as elm

HERE = os.path.dirname(os.path.abspath(__file__))


def draw(path):
    with schemdraw.Drawing(file=path, show=False, dpi=150) as d:
        d.config(fontsize=11, unit=2.4)

        # --- input filter -----------------------------------------------------------------
        elm.Dot(open=True).label('Red Pitaya OUT1\n(±1 V)', loc='left')
        elm.Resistor().right().label('R1\n100 Ω')
        a = elm.Dot()
        elm.Resistor().down().at(a.center).label('R2\n10 kΩ', loc='bottom')
        elm.Ground()
        elm.Line().right(2.0).at(a.center)
        c = elm.Dot()
        elm.Capacitor().down().at(c.center).label('C1\n1 nF', loc='bottom')
        elm.Ground()
        elm.Line().right(1.0).at(c.center)

        # --- amplifier ----------------------------------------------------------------------
        op = elm.Opamp(leads=True).flip().anchor('in2').label('U2\nLM1875', loc='center',
                                                                ofst=(-0.4, 0), fontsize=10)
        top, bot = (op.vd, op.vs) if op.vd[1] > op.vs[1] else (op.vs, op.vd)
        elm.Line().up(0.8).at(top)
        elm.Label().label('+16 V (pin 5)', loc='top')
        elm.Line().down(0.8).at(bot)
        elm.Label().label('−16 V (pin 3)', loc='bottom')

        # feedback divider below the amplifier
        elm.Line().down(1.6).at(op.in1)
        fb = elm.Dot()
        elm.Resistor().down().at(fb.center).label('R7\n1 kΩ', loc='bottom')
        elm.Ground()

        # --- output node, protection --------------------------------------------------------
        elm.Line().right(0.8).at(op.out)
        o1 = elm.Dot()
        elm.Line().right(1.6)
        o2 = elm.Dot()
        elm.Diode().up().at(o2.center).label('D1\n1N5822', loc='bottom')
        elm.Line().up(0.4).label('+16 V', loc='right')
        elm.Diode().down().at(o2.center).reverse().label('D2\nnot fitted', loc='bottom')
        elm.Line().down(0.4).label('−16 V', loc='right')
        elm.Line().right(2.6).at(o2.center)
        o3 = elm.Dot()
        elm.Resistor().down().at(o3.center).label('2.7 Ω\n1 W', loc='bottom')
        elm.Capacitor().down().label('22 nF', loc='bottom')
        elm.Ground()

        # --- load path ------------------------------------------------------------------------
        elm.Resistor().right().at(o3.center).label('R10\n2.2 Ω, 25 W')
        elm.Line().right(0.6)
        elm.Inductor2(loops=4).right().label('load: Helmholtz coil', loc='top').label(
            'R = 17.5 mm, 2 × 12 turns', loc='bottom')
        elm.Line().right(1.4)
        s = elm.Dot()
        elm.Line().right(1.2)
        elm.Dot(open=True).label('Red Pitaya IN1\n0.5 V/A', loc='right')
        elm.Resistor().down().at(s.center).label('R11\n0.5 Ω shunt', loc='bottom')
        elm.Ground().label('star GND', loc='right')

        # --- feedback resistor and MON divider from the output node ---------------------------
        elm.Line().down().at(o1.center).toy(fb.center)
        o1b = elm.Dot()
        elm.Resistor().left().at(o1b.center).tox(fb.center).label('R8\n15 kΩ', loc='bottom')
        elm.Line().down(4.4).at(o1b.center)
        elm.Resistor().right().label('R12\n15 kΩ', loc='bottom')
        m = elm.Dot()
        elm.Line().right(1.2)
        elm.Dot(open=True).label('MON → Red Pitaya IN2\n1/16 of the output', loc='right')
        elm.Resistor().down().at(m.center).label('R13\n1 kΩ', loc='bottom')
        elm.Ground()

        # --- supply -----------------------------------------------------------------------------
        x0, y0 = a.center[0] - 1.0, m.center[1] - 4.5
        elm.Label().at((x0 - 0.5, y0 + 1.6)).label('Supply: NGP824 Ch1 and Ch4 in series',
                                                   loc='right', halign='left')
        top_rail = elm.Dot().at((x0, y0))
        elm.Battery().down().label('Ch1\n16 V', loc='bottom')
        g = elm.Dot().label('star GND', loc='left')
        elm.Battery().down().label('Ch4\n16 V', loc='bottom')
        bot_rail = elm.Dot()
        xs = [x0 + 3.0, x0 + 6.0, x0 + 9.0]
        for (x0_, x1_) in ((x0, xs[0]), (xs[0], xs[1]), (xs[1], xs[2])):
            elm.Line().at((x0_, top_rail.center[1])).to((x1_, top_rail.center[1]))
            elm.Line().at((x0_, g.center[1])).to((x1_, g.center[1]))
            elm.Line().at((x0_, bot_rail.center[1])).to((x1_, bot_rail.center[1]))
        for x, part in zip(xs, ('100n', '470u', 'diode')):
            for y_a, y_b in ((top_rail.center[1], g.center[1]), (g.center[1], bot_rail.center[1])):
                if part == '100n':
                    elm.Capacitor().at((x, y_a)).to((x, y_b)).label('100 nF', loc='bottom')
                elif part == '470u':
                    elm.Capacitor(polar=True).at((x, y_a)).to((x, y_b)).label('470 µF', loc='bottom')
                else:
                    elm.Diode().at((x, y_b)).to((x, y_a)).label(
                        'D3' if y_a == top_rail.center[1] else 'D4', loc='bottom')
                elm.Dot().at((x, y_a))
                elm.Dot().at((x, y_b))
        elm.Line().right(1.5).at((xs[2], top_rail.center[1])).label('+16 V → U2 pin 5, D1', loc='right')
        elm.Line().right(1.5).at((xs[2], g.center[1])).label('GND (star point)', loc='right')
        elm.Line().right(1.5).at((xs[2], bot_rail.center[1])).label('−16 V → U2 pin 3', loc='right')
        elm.Label().at((x0 - 0.5, bot_rail.center[1] - 1.2)).label(
            '100 nF at the chip pins, 470 µF within 2 cm. D3/D4 are reverse-biased normally and '
            'protect the supply if one rail drops out.', loc='right', halign='left', fontsize=9)


if __name__ == '__main__':
    for ext in ('svg', 'png'):
        draw(os.path.join(HERE, f'schematic.{ext}'))
