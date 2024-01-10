/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the Chameleon License found in the
 * LICENSE file in the root directory of this source tree.
 */

import { ArrowUpRight, Menu, LogoGithub } from "@carbon/icons-react";

export type NavContent = {
  title: string;
  description: string;
  showHomeLink?: boolean;
  githubLink?: string;
  navItems: {
    id: string;
    url?: string;
    title: string;
    showArrowIcon?: boolean;
  }[];
};

export type NavProps = {
  position?: "fixed" | "absolute" | "relative";
  variant?: string;
  content: NavContent;
  selected?: string;
  logoIconSrc?: string;
  basePath?: string;
  handleSelect?: (selected: string) => void;
};

export function BasicNavbar({
  selected = "",
  basePath = "/",
  position,

  content,
  logoIconSrc,
}: NavProps) {
  const logoWithLink = () =>
    logoIconSrc ? (
      <div className="flex-shrink-0 hidden md:flex items-center cursor-pointer">
        <a href={basePath}>
          <img className="h-10 w-auto" src={logoIconSrc} alt="" />
        </a>
      </div>
    ) : null;

  const desktopMenuItem = (selected: string, id: string) => {
    return `p-0 m-3 border-b-[1px] bg-transparent rounded-none hover:bg-transparent hover:border-gray-600 focus:border-0 focus:text-primary active:bg-transparent active:text-gray-800 ${
      selected === id ? "border-primary" : "border-transparent"
    }`;
  };

  const getItemLink = (item, className = "") => {
    const url = item.url ? item.url : `${basePath}${item.id}`;
    return (
      <a href={url} className={className}>
        {item.title} {item.showArrowIcon && <ArrowUpRight />}
      </a>
    );
  };

  return (
    <div className={`${position} navbar h-[88px] p-2 md:p-10 z-10 `}>
      <div className="navbar-start lg:flex-3 lg:w-1/2 w-full relative">
        <div className="dropdown absolute start-0">
          <label tabIndex={0} className="btn btn-ghost lg:hidden">
            <Menu className="w-6 h-6" />
          </label>

          {/* Mobile Menu */}
          <ul
            tabIndex={0}
            className={`menu dropdown-content mt-3 p-4 shadow w-[80vw] bg-base-100`}
          >
            {content.navItems &&
              content.navItems.map((item) =>
                content.showHomeLink || item.id !== "home" ? (
                  <li key={item.id} className="font-medium">
                    {getItemLink(item, "border-0")}
                  </li>
                ) : null,
              )}

            {content.githubLink && content.githubLink !== "" && (
              <li>
                <a
                  href={content.githubLink}
                  className="border-0"
                  target="_blank"
                  rel="noreferrer"
                >
                  <LogoGithub className="w-6 h-6" />
                </a>
              </li>
            )}
          </ul>
        </div>
        <div className="mx-3 flex gap-1 lg:w-auto w-full">
          {logoWithLink()}
          <div className="leading-tight text-center w-full lg:text-left lg:w-auto lg:mr-0">
            <div className="lg:text-2xl text-xl font-bold lg:font-bold">
              <a href={basePath} className="border-none">
                {content.title}
              </a>
            </div>
            <div className="text-xs text-gray-600">{content.description}</div>
          </div>
        </div>
      </div>

      {/* Desktop menu */}
      <div className="navbar-center hidden lg:flex lg:flex-1 lg:flex-grow-6 lg:justify-end ">
        <ul className="menu menu-horizontal">
          {content.navItems &&
            content.navItems.map((item) =>
              content.showHomeLink || item.id !== "home" ? (
                <li className="font-medium" key={item.id}>
                  {getItemLink(item, desktopMenuItem(selected, item.id))}
                </li>
              ) : null,
            )}
          {content.githubLink && content.githubLink !== "" && (
            <li>
              <a
                href={content.githubLink}
                target="_blank"
                rel="noreferrer"
                className="no-style rounded-md"
              >
                <LogoGithub className="w-6 h-6" />
              </a>
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}
