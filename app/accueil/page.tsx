'use client'
import Intro from "../ui/accueil/intro"
import Choix from "../ui/accueil/choix"
import About from "../ui/accueil/about"
import Testimonials from "../ui/accueil/testimonial"

export default function Page() {
    return (
        <>
            <Intro />
            <About />
            <Choix />
            <Testimonials />
            
        </>
    )
}