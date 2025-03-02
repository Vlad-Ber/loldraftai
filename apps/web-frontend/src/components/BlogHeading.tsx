"use client";

import React, { ReactNode, useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import { slug } from "github-slugger";

interface HeadingProps {
  level: 1 | 2 | 3 | 4 | 5 | 6;
  children: ReactNode;
  className?: string;
}

export function BlogHeading({ level, children, className = "" }: HeadingProps) {
  const router = useRouter();
  const pathname = usePathname();

  // Get text content from children to generate the slug
  const text = typeof children === "string" ? children : "";
  const headingId = slug(text);

  // Generate the anchor link
  const href = `${pathname}#${headingId}`;

  // Update URL when clicking the heading
  const handleClick = () => {
    router.push(href, { scroll: false });

    // Account for the fixed navbar height (approximately 60px)
    const navbarHeight = 60;
    const element = document.getElementById(headingId);

    if (element) {
      const elementPosition = element.getBoundingClientRect().top;
      const offsetPosition =
        elementPosition + window.pageYOffset - navbarHeight;

      window.scrollTo({
        top: offsetPosition,
        behavior: "smooth",
      });
    }
  };

  const HeadingTag = `h${level}` as keyof JSX.IntrinsicElements;

  useEffect(() => {
    // Check if the URL hash matches this heading's ID
    if (window.location.hash === `#${headingId}`) {
      // Scroll to the element with a slight delay to ensure it's rendered
      // Account for the fixed navbar height
      setTimeout(() => {
        const navbarHeight = 60;
        const element = document.getElementById(headingId);

        if (element) {
          const elementPosition = element.getBoundingClientRect().top;
          const offsetPosition =
            elementPosition + window.pageYOffset - navbarHeight;

          window.scrollTo({
            top: offsetPosition,
            behavior: "smooth",
          });
        }
      }, 100);
    }
  }, [headingId]);

  return (
    <HeadingTag
      id={headingId}
      className={`group cursor-pointer ${className}`}
      onClick={handleClick}
    >
      {children}
    </HeadingTag>
  );
}

export function BlogTableOfContents({
  headings,
}: {
  headings: { text: string; level: number }[];
}) {
  return (
    <nav className="mb-8 p-4 bg-primary/5 rounded-lg">
      <h2 className="text-xl font-semibold mb-2 mt-0">Table of Contents</h2>
      <ul className="space-y-1.5">
        {headings.map((heading, index) => (
          <li
            key={index}
            style={{ marginLeft: `${(heading.level - 2) * 1}rem` }}
          >
            <a
              href={`#${slug(heading.text)}`}
              className="text-primary hover:underline transition-colors"
              onClick={(e) => {
                e.preventDefault();

                // Update URL
                const headingId = slug(heading.text);
                window.history.pushState(null, "", `#${headingId}`);

                // Account for the fixed navbar height (approximately 60px)
                const navbarHeight = 60;
                const element = document.getElementById(headingId);

                if (element) {
                  const elementPosition = element.getBoundingClientRect().top;
                  const offsetPosition =
                    elementPosition + window.pageYOffset - navbarHeight;

                  window.scrollTo({
                    top: offsetPosition,
                    behavior: "smooth",
                  });
                }
              }}
            >
              {heading.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}

export function extractHeadingsFromChildren(
  children: ReactNode
): { text: string; level: number }[] {
  const headings: { text: string; level: number }[] = [];

  // This is a simple implementation that works with common React patterns
  // A more robust solution might need to traverse the React element tree
  React.Children.forEach(children, (child) => {
    if (React.isValidElement(child)) {
      // If the child is a heading component (h1-h6)
      const type = child.type as string | React.JSXElementConstructor<React.ReactNode>;
      if (typeof type === "string" && type.match(/^h[1-6]$/)) {
        const level = parseInt(type.charAt(1));
        const text = React.Children.toArray(child.props.children)
          .filter((c): c is string => typeof c === "string")
          .join("");

        headings.push({ text, level });
      }

      // If the child has children, recursively extract headings
      if (child.props.children) {
        headings.push(...extractHeadingsFromChildren(child.props.children));
      }
    }
  });

  return headings;
}
